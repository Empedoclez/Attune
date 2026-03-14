import Foundation

/// Signal processing options for faithful oscilloscope display.
///
/// Design principles:
/// - Default mode is "faithful": no processing except linear gain
/// - Optional DC block can be enabled for specific use cases
/// - All processing is linear (no compression, limiting, or nonlinear transforms)
public struct SignalProcessingConfig: Sendable {
    /// Linear gain multiplier (default 1.0, range typically 0.1...10.0)
    public var gain: Float
    
    /// Enable DC blocking high-pass filter (default OFF for faithful mode)
    public var dcBlockEnabled: Bool
    
    /// DC block filter cutoff frequency in Hz (when enabled)
    /// Typical range: 10-50 Hz
    public var dcBlockCutoffHz: Float
    
    /// Create default faithful mode configuration (gain only, no DC block)
    public init(
        gain: Float = 1.0,
        dcBlockEnabled: Bool = false,
        dcBlockCutoffHz: Float = 20.0
    ) {
        self.gain = gain
        self.dcBlockEnabled = dcBlockEnabled
        self.dcBlockCutoffHz = dcBlockCutoffHz
    }
    
    /// Default configuration for faithful mode
    public static let faithful = SignalProcessingConfig()
}

/// In-place signal processor for audio samples.
///
/// Applies linear gain and optional DC blocking filter.
/// All operations are in-place to avoid allocations.
public final class SignalProcessor {
    
    // MARK: - Properties
    
    /// Current configuration
    public var config: SignalProcessingConfig
    
    // MARK: - DC Block State
    
    /// DC block filter state for left channel
    private var dcBlockStateL: DCBlockState = DCBlockState()
    
    /// DC block filter state for right channel  
    private var dcBlockStateR: DCBlockState = DCBlockState()
    
    /// Cached filter coefficient
    private var dcBlockAlpha: Float = 0.0
    
    /// Sample rate for filter coefficient calculation
    private var sampleRate: Float = 48000.0
    
    // MARK: - Initialization
    
    /// Create a signal processor with the given configuration.
    public init(config: SignalProcessingConfig = .faithful, sampleRate: Float = 48000.0) {
        self.config = config
        self.sampleRate = sampleRate
        updateFilterCoefficients()
    }
    
    // MARK: - Configuration
    
    /// Update the sample rate (recalculates filter coefficients)
    public func setSampleRate(_ rate: Float) {
        sampleRate = rate
        updateFilterCoefficients()
    }
    
    /// Update configuration
    public func setConfig(_ newConfig: SignalProcessingConfig) {
        let cutoffChanged = config.dcBlockCutoffHz != newConfig.dcBlockCutoffHz
        config = newConfig
        if cutoffChanged {
            updateFilterCoefficients()
        }
    }
    
    // MARK: - Processing
    
    /// Process interleaved stereo samples in-place.
    /// - Parameters:
    ///   - samples: Pointer to interleaved stereo samples [L0, R0, L1, R1, ...]
    ///   - frameCount: Number of stereo frames
    public func process(samples: UnsafeMutablePointer<Float>, frameCount: Int) {
        let gain = config.gain
        let dcBlockEnabled = config.dcBlockEnabled
        
        if dcBlockEnabled {
            processWithDCBlock(samples: samples, frameCount: frameCount, gain: gain)
        } else {
            // Gain only (faithful mode)
            processGainOnly(samples: samples, frameCount: frameCount, gain: gain)
        }
    }
    
    /// Reset filter state (call when starting new audio stream)
    public func reset() {
        dcBlockStateL = DCBlockState()
        dcBlockStateR = DCBlockState()
    }
    
    // MARK: - Private Processing
    
    private func processGainOnly(samples: UnsafeMutablePointer<Float>, frameCount: Int, gain: Float) {
        // Simple gain application - no other processing
        // This is the faithful mode: just scale the signal
        for i in 0..<(frameCount * 2) {
            samples[i] *= gain
        }
    }
    
    private func processWithDCBlock(samples: UnsafeMutablePointer<Float>, frameCount: Int, gain: Float) {
        let alpha = dcBlockAlpha
        
        for i in 0..<frameCount {
            let idx = i * 2
            
            // Left channel
            let inputL = samples[idx]
            let outputL = dcBlockStateL.process(input: inputL, alpha: alpha)
            samples[idx] = outputL * gain
            
            // Right channel
            let inputR = samples[idx + 1]
            let outputR = dcBlockStateR.process(input: inputR, alpha: alpha)
            samples[idx + 1] = outputR * gain
        }
    }
    
    private func updateFilterCoefficients() {
        // Single-pole high-pass filter coefficient
        // alpha = 1 / (1 + 2*pi*fc/fs)
        // Higher alpha = lower cutoff frequency
        let fc = config.dcBlockCutoffHz
        let fs = sampleRate
        let omega = 2.0 * Float.pi * fc / fs
        dcBlockAlpha = 1.0 / (1.0 + omega)
    }
}

// MARK: - DC Block Filter State

/// State for a single-pole DC blocking high-pass filter.
/// Implements: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
fileprivate struct DCBlockState {
    var prevInput: Float = 0.0
    var prevOutput: Float = 0.0
    
    mutating func process(input: Float, alpha: Float) -> Float {
        // High-pass filter: y = alpha * (y_prev + x - x_prev)
        let output = alpha * (prevOutput + input - prevInput)
        prevInput = input
        prevOutput = output
        return output
    }
}

// MARK: - Convenience Extensions

extension SignalProcessor {
    /// Check if any non-faithful processing is enabled
    public var isStrictlyFaithful: Bool {
        !config.dcBlockEnabled && config.gain == 1.0
    }
    
    /// Debug description of current processing chain
    public var processingDescription: String {
        var parts: [String] = []
        
        if config.gain != 1.0 {
            parts.append("Gain: \(String(format: "%.2f", config.gain))x")
        }
        
        if config.dcBlockEnabled {
            parts.append("DC Block: \(Int(config.dcBlockCutoffHz)) Hz")
        }
        
        if parts.isEmpty {
            return "Processing: None (faithful mode)"
        }
        
        return "Processing: " + parts.joined(separator: ", ")
    }
}
