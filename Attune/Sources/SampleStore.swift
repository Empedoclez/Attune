import Atomics
import Foundation

/// Thread-safe, lock-free ring buffer for stereo audio samples.
///
/// Design:
/// - Single producer (audio callback) writes interleaved stereo Float32 frames
/// - Multiple consumers can read snapshots without blocking
/// - Power-of-two capacity for efficient wraparound
/// - Atomic write index with release/acquire ordering
///
/// Frame layout: [L0, R0, L1, R1, ...] where each frame is 2 Float32 values
public final class SampleStore: @unchecked Sendable {
    
    /// Number of frames (stereo pairs) the buffer can hold. Always a power of two.
    public let capacityFrames: Int
    
    /// Mask for efficient wraparound: (index & mask) instead of (index % capacity)
    private let indexMask: Int
    
    /// Interleaved stereo samples buffer [L0, R0, L1, R1, ...]
    /// Size = capacityFrames * 2
    private let buffer: UnsafeMutableBufferPointer<Float>
    
    /// Monotonically increasing frame counter. Represents total frames ever written.
    /// Readers use this to compute the valid range of readable frames.
    private let _writeIndexFrames: ManagedAtomic<Int64>
    
    /// Returns the current write index (total frames written since start).
    public var writeIndexFrames: Int64 {
        _writeIndexFrames.load(ordering: .acquiring)
    }
    
    /// Initialize the sample store with a given capacity.
    /// - Parameter capacityFrames: Number of stereo frames to hold. Will be rounded up to power of two.
    public init(capacityFrames: Int) {
        // Round up to next power of two for efficient modulo
        let capacity = Self.nextPowerOfTwo(capacityFrames)
        self.capacityFrames = capacity
        self.indexMask = capacity - 1
        
        // Allocate buffer for interleaved stereo (2 samples per frame)
        let sampleCount = capacity * 2
        self.buffer = .allocate(capacity: sampleCount)
        buffer.initialize(repeating: 0.0)
        
        self._writeIndexFrames = ManagedAtomic<Int64>(0)
    }
    
    deinit {
        buffer.deallocate()
    }
    
    // MARK: - Producer API (Audio Callback - RT Safe)
    
    /// Write interleaved stereo frames to the buffer.
    /// This is the only write method and must be called from a single producer thread.
    /// - Parameters:
    ///   - frames: Pointer to interleaved stereo samples [L0, R0, L1, R1, ...]
    ///   - count: Number of frames (stereo pairs) to write
    /// - Note: This method performs no allocations and acquires no locks.
    public func write(frames: UnsafePointer<Float>, count: Int) {
        let currentWrite = _writeIndexFrames.load(ordering: .relaxed)
        
        for i in 0..<count {
            let writePos = Int((currentWrite + Int64(i)) & Int64(indexMask))
            let sampleOffset = writePos * 2
            buffer[sampleOffset] = frames[i * 2]       // Left
            buffer[sampleOffset + 1] = frames[i * 2 + 1] // Right
        }
        
        // Publish the new write position with release ordering
        _writeIndexFrames.store(currentWrite + Int64(count), ordering: .releasing)
    }
    
    // MARK: - Consumer API: Snapshot Reader (Renderer)
    
    /// Copy the most recent N frames into a destination buffer.
    /// Used by the renderer to get a window of samples for display.
    /// - Parameters:
    ///   - count: Number of frames to copy
    ///   - destination: Buffer to copy frames into (must hold count * 2 Float values)
    /// - Returns: Actual number of frames copied (may be less if buffer doesn't have enough data)
    public func copyLatest(frames count: Int, into destination: UnsafeMutablePointer<Float>) -> Int {
        let currentWrite = _writeIndexFrames.load(ordering: .acquiring)
        
        // Clamp to available data
        let available = min(Int64(count), currentWrite, Int64(capacityFrames))
        let actualCount = Int(available)
        
        if actualCount == 0 { return 0 }
        
        // Calculate start frame index (in absolute terms, then wrap)
        let startFrame = currentWrite - Int64(actualCount)
        
        for i in 0..<actualCount {
            let readPos = Int((startFrame + Int64(i)) & Int64(indexMask))
            let sampleOffset = readPos * 2
            destination[i * 2] = buffer[sampleOffset]       // Left
            destination[i * 2 + 1] = buffer[sampleOffset + 1] // Right
        }
        
        return actualCount
    }
    
    /// Copy the most recent N frames as Float2 (L, R) pairs for GPU consumption.
    /// - Parameters:
    ///   - count: Number of frames to copy
    ///   - destination: Buffer to copy into (x = L, y = R)
    /// - Returns: Actual number of frames copied
    public func copyLatestAsFloat2(frames count: Int, into destination: UnsafeMutablePointer<SIMD2<Float>>) -> Int {
        let currentWrite = _writeIndexFrames.load(ordering: .acquiring)
        
        let available = min(Int64(count), currentWrite, Int64(capacityFrames))
        let actualCount = Int(available)
        
        if actualCount == 0 { return 0 }
        
        let startFrame = currentWrite - Int64(actualCount)
        
        for i in 0..<actualCount {
            let readPos = Int((startFrame + Int64(i)) & Int64(indexMask))
            let sampleOffset = readPos * 2
            destination[i] = SIMD2(buffer[sampleOffset], buffer[sampleOffset + 1])
        }
        
        return actualCount
    }
    
    // MARK: - Consumer API: Sequential Drain (Recorder)
    
    /// A sequential reader that tracks its own position for continuous consumption.
    /// Used by the recorder to read all samples without gaps or duplicates.
    public final class SequentialReader {
        private let store: SampleStore
        
        /// The next frame index this reader expects to read
        private var nextReadFrame: Int64 = 0
        
        /// True if this reader has fallen behind and lost data
        public private(set) var hasOverflowed: Bool = false

        /// Total number of overflow events observed for this reader
        public private(set) var overflowCount: Int = 0
        
        init(store: SampleStore) {
            self.store = store
        }
        
        /// Drain available frames into destination buffer.
        /// - Parameters:
        ///   - destination: Buffer to write frames into
        ///   - maxFrames: Maximum frames to read
        /// - Returns: Number of frames actually read
        public func drain(into destination: UnsafeMutablePointer<Float>, maxFrames: Int) -> Int {
            let currentWrite = store._writeIndexFrames.load(ordering: .acquiring)
            
            // Check for overflow: reader is more than capacity behind
            let lag = currentWrite - nextReadFrame
            if lag > Int64(store.capacityFrames) {
                // Overflow detected - resync to oldest available data
                hasOverflowed = true
                overflowCount += 1
                nextReadFrame = currentWrite - Int64(store.capacityFrames)
            }
            
            // Calculate how many frames we can read
            let availableFrames = min(Int64(maxFrames), currentWrite - nextReadFrame)
            let readCount = Int(max(0, availableFrames))
            
            if readCount == 0 { return 0 }
            
            // Copy frames
            for i in 0..<readCount {
                let readPos = Int((nextReadFrame + Int64(i)) & Int64(store.indexMask))
                let sampleOffset = readPos * 2
                destination[i * 2] = store.buffer[sampleOffset]
                destination[i * 2 + 1] = store.buffer[sampleOffset + 1]
            }
            
            nextReadFrame += Int64(readCount)
            return readCount
        }
        
        /// Reset overflow flag after handling the overflow condition
        public func acknowledgeOverflow() {
            hasOverflowed = false
        }
        
        /// Sync to current write position (discard unread data)
        public func resync() {
            nextReadFrame = store._writeIndexFrames.load(ordering: .acquiring)
            hasOverflowed = false
        }
    }
    
    /// Create a new sequential reader for this store.
    /// The reader starts at the current write position.
    public func makeSequentialReader() -> SequentialReader {
        let reader = SequentialReader(store: self)
        reader.resync()
        return reader
    }
    
    // MARK: - Helpers
    
    private static func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        if n & (n - 1) == 0 { return n } // Already power of two
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        return v + 1
    }
}
