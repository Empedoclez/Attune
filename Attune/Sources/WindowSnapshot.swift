import Foundation
import simd

/// Configuration for audio window snapshots.
public struct WindowConfig: Sendable {
    /// Canonical window duration at 48kHz reference (16384/48000 ≈ 0.341333s)
    public static let canonicalDurationSeconds: Double = 16384.0 / 48000.0
    
    /// Sample rate of the audio source
    public let sampleRate: Double
    
    /// Number of frames in the window (computed from sample rate)
    public let frameCount: Int
    
    /// Actual duration in seconds for this sample rate
    public var durationSeconds: Double {
        Double(frameCount) / sampleRate
    }
    
    /// Create a window configuration for a given sample rate.
    /// - Parameter sampleRate: The sample rate in Hz (e.g., 44100, 48000, 96000)
    public init(sampleRate: Double) {
        self.sampleRate = sampleRate
        // N = round(Twindow * fs)
        self.frameCount = Int(round(Self.canonicalDurationSeconds * sampleRate))
    }
    
    /// Create with explicit frame count (for testing)
    public init(sampleRate: Double, frameCount: Int) {
        self.sampleRate = sampleRate
        self.frameCount = frameCount
    }
}

/// Manages window snapshots from a SampleStore for GPU consumption.
///
/// Design:
/// - Pre-allocates point buffer to avoid allocations during updates
/// - Converts interleaved stereo (L, R) to SIMD2<Float> (x=L, y=R) for GPU
/// - Thread-safe: can be updated from render thread
public final class WindowSnapshot {
    
    // MARK: - Properties
    
    /// The sample store to read from
    public let sampleStore: SampleStore
    
    /// Current window configuration
    public private(set) var config: WindowConfig
    
    /// Pre-allocated point buffer for GPU consumption
    /// Contains SIMD2<Float> where x=Left, y=Right
    private var pointBuffer: UnsafeMutablePointer<SIMD2<Float>>
    
    /// Current point buffer capacity (frames)
    private var pointBufferCapacity: Int
    
    /// Number of valid points after last update
    public private(set) var validPointCount: Int = 0
    
    /// Timestamp of last update
    public private(set) var lastUpdateTime: CFAbsoluteTime = 0
    
    // MARK: - Initialization
    
    /// Create a window snapshot manager.
    /// - Parameters:
    ///   - sampleStore: The sample store to read from
    ///   - sampleRate: Initial sample rate (can be updated later)
    public init(sampleStore: SampleStore, sampleRate: Double = 48000.0) {
        self.sampleStore = sampleStore
        self.config = WindowConfig(sampleRate: sampleRate)
        
        // Allocate initial buffer
        let capacity = config.frameCount
        self.pointBufferCapacity = capacity
        self.pointBuffer = .allocate(capacity: capacity)
        self.pointBuffer.initialize(repeating: .zero, count: capacity)
    }
    
    deinit {
        pointBuffer.deallocate()
    }
    
    // MARK: - Configuration
    
    /// Update the sample rate (e.g., when audio device changes).
    /// This will reallocate the point buffer if needed.
    /// - Parameter sampleRate: New sample rate in Hz
    public func updateSampleRate(_ sampleRate: Double) {
        let newConfig = WindowConfig(sampleRate: sampleRate)
        
        // Only reallocate if size changed significantly
        if newConfig.frameCount != config.frameCount {
            reallocateBuffer(newCapacity: newConfig.frameCount)
        }
        
        config = newConfig
    }
    
    // MARK: - Snapshot Operations
    
    /// Update the point buffer with the latest audio window.
    /// Call this once per frame from the render thread.
    /// - Returns: Number of valid points in the buffer
    @discardableResult
    public func update() -> Int {
        let frameCount = config.frameCount
        
        // Ensure buffer is large enough
        if frameCount > pointBufferCapacity {
            reallocateBuffer(newCapacity: frameCount)
        }
        
        // Copy latest frames as Float2 directly (no intermediate buffer)
        let copied = sampleStore.copyLatestAsFloat2(
            frames: frameCount,
            into: pointBuffer
        )
        
        validPointCount = copied
        lastUpdateTime = CFAbsoluteTimeGetCurrent()
        
        return copied
    }
    
    /// Get the current point buffer for GPU consumption.
    /// - Returns: Tuple of (pointer, count) for the valid points
    public func getPoints() -> (UnsafePointer<SIMD2<Float>>, Int) {
        return (UnsafePointer(pointBuffer), validPointCount)
    }
    
    /// Get the point buffer as a typed buffer pointer.
    public var points: UnsafeBufferPointer<SIMD2<Float>> {
        UnsafeBufferPointer(start: pointBuffer, count: validPointCount)
    }
    
    // MARK: - Private
    
    private func reallocateBuffer(newCapacity: Int) {
        pointBuffer.deallocate()
        pointBufferCapacity = newCapacity
        pointBuffer = .allocate(capacity: newCapacity)
        pointBuffer.initialize(repeating: .zero, count: newCapacity)
        validPointCount = 0
    }
}

// MARK: - Convenience Extensions

extension WindowSnapshot {
    /// Check if the snapshot has valid data
    public var hasValidData: Bool {
        validPointCount > 0
    }
    
    /// Estimated time since data was captured (for latency monitoring)
    public var dataStalenessSeconds: Double {
        CFAbsoluteTimeGetCurrent() - lastUpdateTime
    }
    
    /// Debug description of current state
    public var debugDescription: String {
        """
        WindowSnapshot:
          SampleRate: \(config.sampleRate) Hz
          FrameCount: \(config.frameCount)
          Duration: \(String(format: "%.3f", config.durationSeconds))s
          ValidPoints: \(validPointCount)
          Staleness: \(String(format: "%.1f", dataStalenessSeconds * 1000))ms
        """
    }
}
