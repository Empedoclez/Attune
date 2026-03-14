import Foundation
import Atomics
import os

/// Parameters for the oscilloscope renderer.
/// Thread-safe: UI writes, renderer reads a snapshot once per frame.
public struct ScopeParams: Sendable {
    /// Input gain (linear multiplier). Default: 1.0
    public var gain: Float
    
    /// Post-map zoom/scale factor. Default: 1.0
    public var zoom: Float
    
    /// Beam thickness in pixels (Gaussian sigma). Default: 1.2
    public var beamSigmaPx: Float
    
    /// Persistence/decay time constant in seconds. Default: 0.6
    public var tauSeconds: Float
    
    /// Beam intensity/brightness scalar. Default: 1.0
    public var intensity: Float
    
    /// Bloom effect amount (0.0 = none, 1.0 = full). Default: 0.35
    public var bloomAmount: Float
    
    /// Bloom blur radius in pixels. Default: 8.0
    public var bloomSigmaPx: Float
    
    /// Phosphor hue (0.0-1.0). Default: 0.35 (greenish)
    public var hue: Float
    
    /// When true, freeze display (no new audio, no decay). Default: false
    public var freeze: Bool
    
    /// When true, compensate for beam travel speed. Default: false (faithful mode)
    public var dwellCompensation: Bool
    
    /// Create default parameters suitable for faithful oscilloscope display.
    public init(
        gain: Float = 1.0,
        zoom: Float = 1.0,
        beamSigmaPx: Float = 1.2,
        tauSeconds: Float = 0.6,
        intensity: Float = 1.0,
        bloomAmount: Float = 0.35,
        bloomSigmaPx: Float = 8.0,
        hue: Float = 0.35,
        freeze: Bool = false,
        dwellCompensation: Bool = false
    ) {
        self.gain = gain
        self.zoom = zoom
        self.beamSigmaPx = beamSigmaPx
        self.tauSeconds = tauSeconds
        self.intensity = intensity
        self.bloomAmount = bloomAmount
        self.bloomSigmaPx = bloomSigmaPx
        self.hue = hue
        self.freeze = freeze
        self.dwellCompensation = dwellCompensation
    }
    
    /// Default parameters for v0 faithful mode.
    public static let `default` = ScopeParams()
}

/// Thread-safe container for ScopeParams.
/// Uses os_unfair_lock for minimal overhead.
/// UI thread writes, render thread reads snapshot once per frame.
public final class ScopeParamsStore: @unchecked Sendable {
    private var _params: ScopeParams
    private var _lock = os_unfair_lock()
    
    public init(params: ScopeParams = .default) {
        self._params = params
    }
    
    /// Read the current parameters (thread-safe snapshot).
    public func snapshot() -> ScopeParams {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        return _params
    }
    
    /// Update parameters (thread-safe).
    public func update(_ params: ScopeParams) {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        _params = params
    }
    
    /// Update a single parameter via closure (thread-safe).
    public func modify(_ block: (inout ScopeParams) -> Void) {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        block(&_params)
    }
    
    // MARK: - Convenience accessors for UI binding
    
    public var gain: Float {
        get { snapshot().gain }
        set { modify { $0.gain = newValue } }
    }
    
    public var zoom: Float {
        get { snapshot().zoom }
        set { modify { $0.zoom = newValue } }
    }
    
    public var beamSigmaPx: Float {
        get { snapshot().beamSigmaPx }
        set { modify { $0.beamSigmaPx = newValue } }
    }
    
    public var tauSeconds: Float {
        get { snapshot().tauSeconds }
        set { modify { $0.tauSeconds = newValue } }
    }
    
    public var intensity: Float {
        get { snapshot().intensity }
        set { modify { $0.intensity = newValue } }
    }
    
    public var bloomAmount: Float {
        get { snapshot().bloomAmount }
        set { modify { $0.bloomAmount = newValue } }
    }
    
    public var bloomSigmaPx: Float {
        get { snapshot().bloomSigmaPx }
        set { modify { $0.bloomSigmaPx = newValue } }
    }
    
    public var hue: Float {
        get { snapshot().hue }
        set { modify { $0.hue = newValue } }
    }
    
    public var freeze: Bool {
        get { snapshot().freeze }
        set { modify { $0.freeze = newValue } }
    }
    
    public var dwellCompensation: Bool {
        get { snapshot().dwellCompensation }
        set { modify { $0.dwellCompensation = newValue } }
    }
}
