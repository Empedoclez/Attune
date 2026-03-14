import AVFoundation
import Foundation

/// Error types for iOS audio capture operations.
public enum AudioCaptureError: Error, LocalizedError {
    case microphonePermissionDenied
    case monoDevice(channelCount: Int)
    case engineStartFailed(Error)
    case noInputNode

    public var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone access denied. Please enable in Settings → Privacy → Microphone."
        case .monoDevice(let channels):
            return """
                Stereo input required. Device has only \(channels) channel(s).
                Please connect a stereo audio source.
                """
        case .engineStartFailed(let error):
            return "Failed to start audio engine: \(error.localizedDescription)"
        case .noInputNode:
            return "No audio input available on this device."
        }
    }
}

/// Information about the current audio input.
public struct AudioDeviceInfo: Sendable {
    public let name: String
    public let channelCount: Int
    public let sampleRate: Double
}

/// AVAudioEngine-based stereo capture for iOS that feeds into a SampleStore.
///
/// Design:
/// - Uses AVAudioEngine input node tap for audio capture
/// - Configures AVAudioSession for lowest possible latency
/// - Requests stereo input when available
/// - Writes interleaved Float32 stereo to SampleStore
///
/// On iPhone, the built-in microphones (bottom + top/front) provide
/// a stereo field when configured correctly. The spatial separation
/// (~13-14cm on Pro models) gives meaningful phase information for
/// voice at arm's length.
public final class AudioCapture {

    // MARK: - Properties

    /// The sample store to write captured audio into
    public let sampleStore: SampleStore

    /// Current device information (available after start)
    public private(set) var deviceInfo: AudioDeviceInfo?

    /// Whether capture is currently running
    public private(set) var isRunning: Bool = false

    /// Last error if any operation failed
    public private(set) var lastError: AudioCaptureError?

    // MARK: - Private Properties

    private let engine = AVAudioEngine()

    /// Pre-allocated interleave buffer to avoid allocations in the tap callback
    private var interleaveBuffer: UnsafeMutablePointer<Float>?
    private var interleaveBufferCapacity: Int = 0

    // MARK: - Initialization

    /// Create an audio capture instance.
    /// - Parameter sampleStore: The sample store to write captured audio into
    public init(sampleStore: SampleStore) {
        self.sampleStore = sampleStore
    }

    deinit {
        stop()
        interleaveBuffer?.deallocate()
    }

    // MARK: - Lifecycle

    /// Request microphone permission, then start capture.
    /// Call this from a context that can handle the async permission flow.
    public func requestPermissionAndStart() async throws {
        // Request permission
        let granted: Bool
        if #available(iOS 17.0, *) {
            granted = await AVAudioApplication.requestRecordPermission()
        } else {
            granted = await withCheckedContinuation { continuation in
                AVAudioSession.sharedInstance().requestRecordPermission { result in
                    continuation.resume(returning: result)
                }
            }
        }

        guard granted else {
            let error = AudioCaptureError.microphonePermissionDenied
            lastError = error
            throw error
        }

        try start()
    }

    /// Start audio capture. Assumes microphone permission is already granted.
    /// - Throws: AudioCaptureError if configuration or start fails
    public func start() throws {
        guard !isRunning else { return }
        lastError = nil

        let session = AVAudioSession.sharedInstance()

        // Configure session for lowest latency stereo input
        try session.setCategory(
            .playAndRecord,
            mode: .measurement,
            options: [.defaultToSpeaker, .allowBluetooth]
        )

        // Request minimum buffer duration for lowest latency
        // 256 frames / 48000 Hz ≈ 5.3ms
        try session.setPreferredIOBufferDuration(256.0 / 48000.0)

        // Request stereo input
        if session.isInputAvailable {
            if let inputs = session.availableInputs {
                // Prefer built-in mic for consistent stereo behavior
                let builtIn = inputs.first { $0.portType == .builtInMic }
                let selectedInput = builtIn ?? inputs.first

                if let input = selectedInput {
                    try session.setPreferredInput(input)

                    // Request stereo if the data source supports it
                    if let dataSources = input.dataSources,
                       let stereoSource = dataSources.first(where: {
                           $0.supportedPolarPatterns?.contains(.stereo) == true
                       }) {
                        try stereoSource.setPreferredPolarPattern(.stereo)
                        try input.setPreferredDataSource(stereoSource)
                    }
                }
            }

            // Request 2 input channels
            try session.setPreferredInputNumberOfChannels(2)
        }

        try session.setActive(true)

        let inputNode = engine.inputNode
        let hwFormat = inputNode.inputFormat(forBus: 0)

        guard hwFormat.channelCount >= 1 else {
            throw AudioCaptureError.noInputNode
        }

        // Store device info
        let sampleRate = hwFormat.sampleRate
        let channelCount = Int(hwFormat.channelCount)
        deviceInfo = AudioDeviceInfo(
            name: session.currentRoute.inputs.first?.portName ?? "iPhone Microphone",
            channelCount: channelCount,
            sampleRate: sampleRate
        )

        // Create the format we want: float32, at the hardware sample rate
        // We handle mono-to-stereo duplication in the tap if needed
        let busFormat = inputNode.outputFormat(forBus: 0)

        // Pre-allocate interleave buffer (max expected: 4096 frames * 2 channels)
        let maxFrames = 8192
        if interleaveBuffer == nil || interleaveBufferCapacity < maxFrames * 2 {
            interleaveBuffer?.deallocate()
            interleaveBufferCapacity = maxFrames * 2
            interleaveBuffer = .allocate(capacity: interleaveBufferCapacity)
        }

        // Install tap on input node
        // The tap callback runs on a real-time audio thread
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: busFormat) {
            [weak self] buffer, _ in
            guard let self = self,
                  let interleaveBuffer = self.interleaveBuffer else { return }

            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0, frameCount * 2 <= self.interleaveBufferCapacity else { return }

            let channelCount = Int(buffer.format.channelCount)

            if buffer.format.isInterleaved {
                // Already interleaved — copy directly
                if let channelData = buffer.floatChannelData?[0] {
                    if channelCount >= 2 {
                        // Stereo interleaved: copy as-is
                        memcpy(interleaveBuffer, channelData, frameCount * 2 * MemoryLayout<Float>.size)
                    } else {
                        // Mono interleaved: duplicate to stereo
                        for i in 0..<frameCount {
                            interleaveBuffer[i * 2] = channelData[i]
                            interleaveBuffer[i * 2 + 1] = channelData[i]
                        }
                    }
                }
            } else {
                // Non-interleaved (standard AVAudioPCMBuffer layout)
                if channelCount >= 2,
                   let ch0 = buffer.floatChannelData?[0],
                   let ch1 = buffer.floatChannelData?[1] {
                    // Stereo: interleave L and R
                    for i in 0..<frameCount {
                        interleaveBuffer[i * 2] = ch0[i]
                        interleaveBuffer[i * 2 + 1] = ch1[i]
                    }
                } else if let ch0 = buffer.floatChannelData?[0] {
                    // Mono: duplicate to both channels
                    for i in 0..<frameCount {
                        interleaveBuffer[i * 2] = ch0[i]
                        interleaveBuffer[i * 2 + 1] = ch0[i]
                    }
                }
            }

            // Write interleaved stereo to sample store (lock-free)
            self.sampleStore.write(frames: interleaveBuffer, count: frameCount)
        }

        // Start the engine
        do {
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            throw AudioCaptureError.engineStartFailed(error)
        }

        isRunning = true
    }

    /// Stop audio capture.
    public func stop() {
        guard isRunning else { return }

        engine.inputNode.removeTap(onBus: 0)
        engine.stop()

        try? AVAudioSession.sharedInstance().setActive(false,
            options: .notifyOthersOnDeactivation)

        isRunning = false
    }
}
