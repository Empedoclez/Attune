import Foundation
import MetalKit
import MetalPerformanceShaders
import QuartzCore
import SwiftUI

struct MetalScopeView: UIViewRepresentable {
    let paramsStore: ScopeParamsStore
    let sampleStore: SampleStore?
    let sampleRate: Double?

    init(paramsStore: ScopeParamsStore,
         sampleStore: SampleStore?,
         sampleRate: Double?) {
        self.paramsStore = paramsStore
        self.sampleStore = sampleStore
        self.sampleRate = sampleRate
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        private struct DecayParams {
            var decay: Float
        }

        private struct BeamParams {
            var viewportSize: SIMD2<Float>
            var sigma: Float
            var intensity: Float
            var hue: Float
            var dwellCompensation: Float
            var _pad: SIMD2<Float> = .zero
        }

        private struct CompositeParams {
            var bloomAmount: Float
            var exposure: Float
            var gamma: Float
            var _pad: Float = 0
        }

        private var commandQueue: MTLCommandQueue?
        private var beamPipelineState: MTLRenderPipelineState?
        private var downsamplePipelineState: MTLRenderPipelineState?
        private var compositePipelineState: MTLRenderPipelineState?
        private var decayPipelineState: MTLComputePipelineState?
        private var pointsBuffer: MTLBuffer?
        private var pointCount: Int = 0
        private var accumTexture: MTLTexture?
        private var bloomTexture: MTLTexture?
        private var bloomTempTexture: MTLTexture?
        private var compositeSampler: MTLSamplerState?
        private var bloomKernel: MPSImageGaussianBlur?
        private var bloomKernelSigma: Float = -1
        private var lastFrameTime: CFTimeInterval?
        private var needsClear = true
        private let paramsStore: ScopeParamsStore
        private let sampleStore: SampleStore?

        private var windowSnapshot: WindowSnapshot?
        private var desiredSampleRate: Double?
        private var pointsBufferCapacity: Int = 0

        init(paramsStore: ScopeParamsStore, sampleStore: SampleStore?, sampleRate: Double?) {
            self.paramsStore = paramsStore
            self.sampleStore = sampleStore
            self.desiredSampleRate = sampleRate
            super.init()
            if let sampleStore = sampleStore {
                let initialRate = sampleRate ?? 48_000.0
                self.windowSnapshot = WindowSnapshot(sampleStore: sampleStore, sampleRate: initialRate)
                self.pointsBufferCapacity = self.windowSnapshot?.config.frameCount ?? 0
            }
        }

        func configure(view: MTKView) {
            guard let device = view.device else { return }
            commandQueue = device.makeCommandQueue()
            view.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.03, alpha: 1.0)
            view.colorPixelFormat = .bgra8Unorm
            view.preferredFramesPerSecond = 120 // Use ProMotion if available
            view.enableSetNeedsDisplay = false
            view.isPaused = false
            view.framebufferOnly = true
            buildPipeline(device: device, view: view)
        }

        func updateSampleRate(_ sampleRate: Double?) {
            desiredSampleRate = sampleRate
        }

        private func buildPipeline(device: MTLDevice, view: MTKView) {
            let source = """
            #include <metal_stdlib>
            using namespace metal;

            struct BeamParams {
                float2 viewportSize;
                float sigma;
                float intensity;
                float hue;
                float dwellCompensation;
                float2 _pad;
            };

            struct BeamOut {
                float4 position [[position]];
                float distPx;
                float intensityScale;
            };

            float3 hue_to_rgb(float h) {
                float3 k = float3(0.0, 2.0 / 3.0, 1.0 / 3.0);
                float3 p = abs(fract(h + k) * 6.0 - 3.0);
                return clamp(p - 1.0, 0.0, 1.0);
            }

            vertex BeamOut v_beam(uint vid [[vertex_id]],
                                  uint iid [[instance_id]],
                                  const device float2 *points [[buffer(0)]],
                                  constant BeamParams &params [[buffer(1)]]) {
                float2 p0 = points[iid];
                float2 p1 = points[iid + 1];
                float2 dir = p1 - p0;
                float2 dirPixel = float2(dir.x * params.viewportSize.x,
                                         dir.y * params.viewportSize.y);
                float lenPx = length(dirPixel);
                float2 dirPx = (lenPx > 0.0) ? (dirPixel / lenPx) : float2(1.0, 0.0);

                float2 normalPx = float2(-dirPx.y, dirPx.x);
                float2 normalNDC = float2(normalPx.x / params.viewportSize.x * 2.0,
                                          normalPx.y / params.viewportSize.y * 2.0);

                float halfWidthPx = max(params.sigma * 3.0, 1.0);
                float side = (vid & 1) == 0 ? -1.0 : 1.0;
                float along = (vid >= 2) ? 1.0 : 0.0;

                float2 base = mix(p0, p1, along);
                float2 offset = normalNDC * halfWidthPx * side;

                float dwellScale = clamp(1.0 / max(lenPx, 1.0), 0.25, 4.0);
                float intensityScale = mix(1.0, dwellScale, params.dwellCompensation);

                BeamOut out;
                out.position = float4(base + offset, 0.0, 1.0);
                out.distPx = side * halfWidthPx;
                out.intensityScale = intensityScale;
                return out;
            }

            fragment float4 f_beam(BeamOut in [[stage_in]],
                                   constant BeamParams &params [[buffer(1)]]) {
                float sigma = max(params.sigma, 0.001);
                float g = exp(-(in.distPx * in.distPx) / (2.0 * sigma * sigma));
                float intensity = params.intensity * in.intensityScale * g;
                float3 color = hue_to_rgb(params.hue);
                return float4(color * intensity, intensity);
            }

            struct FullscreenOut {
                float4 position [[position]];
                float2 uv;
            };

            vertex FullscreenOut v_fullscreen(uint vid [[vertex_id]]) {
                float2 positions[3] = {
                    float2(-1.0, -1.0),
                    float2( 3.0, -1.0),
                    float2(-1.0,  3.0)
                };
                float2 uvs[3] = {
                    float2(0.0, 0.0),
                    float2(2.0, 0.0),
                    float2(0.0, 2.0)
                };
                FullscreenOut out;
                out.position = float4(positions[vid], 0.0, 1.0);
                out.uv = uvs[vid];
                return out;
            }

            struct CompositeParams {
                float bloomAmount;
                float exposure;
                float gamma;
                float _pad;
            };

            fragment float4 f_downsample(FullscreenOut in [[stage_in]],
                                         texture2d<float> accum [[texture(0)]],
                                         sampler samp [[sampler(0)]]) {
                float2 uv = clamp(in.uv, float2(0.0), float2(1.0));
                return accum.sample(samp, uv);
            }

            fragment float4 f_composite(FullscreenOut in [[stage_in]],
                                        texture2d<float> accum [[texture(0)]],
                                        texture2d<float> bloom [[texture(1)]],
                                        sampler samp [[sampler(0)]],
                                        constant CompositeParams &params [[buffer(0)]]) {
                float2 uv = clamp(in.uv, float2(0.0), float2(1.0));
                float3 color = accum.sample(samp, uv).rgb;
                float3 glow = bloom.sample(samp, uv).rgb * params.bloomAmount;
                float3 combined = max(color + glow, float3(0.0));
                float exposure = max(params.exposure, 0.001);
                float3 mapped = 1.0 - exp(-combined * exposure);
                float gamma = max(params.gamma, 0.001);
                float3 corrected = pow(mapped, float3(1.0 / gamma));
                return float4(saturate(corrected), 1.0);
            }

            struct DecayParams {
                float decay;
            };

            kernel void decay_kernel(texture2d<float, access::read_write> tex [[texture(0)]],
                                     constant DecayParams &params [[buffer(0)]],
                                     uint2 gid [[thread_position_in_grid]]) {
                if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) {
                    return;
                }
                float4 color = tex.read(gid);
                tex.write(color * params.decay, gid);
            }
            """

            do {
                let library = try device.makeLibrary(source: source, options: nil)
                let beamDescriptor = MTLRenderPipelineDescriptor()
                beamDescriptor.vertexFunction = library.makeFunction(name: "v_beam")
                beamDescriptor.fragmentFunction = library.makeFunction(name: "f_beam")
                beamDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
                beamDescriptor.colorAttachments[0].isBlendingEnabled = true
                beamDescriptor.colorAttachments[0].rgbBlendOperation = .add
                beamDescriptor.colorAttachments[0].alphaBlendOperation = .add
                beamDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
                beamDescriptor.colorAttachments[0].destinationRGBBlendFactor = .one
                beamDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
                beamDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .one
                beamPipelineState = try device.makeRenderPipelineState(descriptor: beamDescriptor)

                let downsampleDescriptor = MTLRenderPipelineDescriptor()
                downsampleDescriptor.vertexFunction = library.makeFunction(name: "v_fullscreen")
                downsampleDescriptor.fragmentFunction = library.makeFunction(name: "f_downsample")
                downsampleDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
                downsamplePipelineState = try device.makeRenderPipelineState(descriptor: downsampleDescriptor)

                let compositeDescriptor = MTLRenderPipelineDescriptor()
                compositeDescriptor.vertexFunction = library.makeFunction(name: "v_fullscreen")
                compositeDescriptor.fragmentFunction = library.makeFunction(name: "f_composite")
                compositeDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
                compositePipelineState = try device.makeRenderPipelineState(descriptor: compositeDescriptor)

                let samplerDescriptor = MTLSamplerDescriptor()
                samplerDescriptor.minFilter = .linear
                samplerDescriptor.magFilter = .linear
                compositeSampler = device.makeSamplerState(descriptor: samplerDescriptor)

                if let decayFunction = library.makeFunction(name: "decay_kernel") {
                    decayPipelineState = try device.makeComputePipelineState(function: decayFunction)
                }
            } catch {
                print("Metal pipeline build failed: \(error)")
            }
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            ensureTextures(view: view)
        }

        func draw(in view: MTKView) {
            guard let commandQueue,
                  let drawable = view.currentDrawable,
                  let renderPass = view.currentRenderPassDescriptor
            else { return }

            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

            ensureTextures(view: view)

            let now = CACurrentMediaTime()
            let dt = Float(lastFrameTime.map { now - $0 } ?? (1.0 / 60.0))
            lastFrameTime = now
            let params = paramsStore.snapshot()

            // Read live audio samples
            if let windowSnapshot,
               let device = view.device,
               !params.freeze {
                if let desiredSampleRate, desiredSampleRate > 0,
                   desiredSampleRate != windowSnapshot.config.sampleRate {
                    windowSnapshot.updateSampleRate(desiredSampleRate)
                }

                let frameCount = windowSnapshot.update()
                if frameCount > 1 {
                    let byteCount = MemoryLayout<SIMD2<Float>>.stride * frameCount
                    if pointsBuffer == nil || pointsBufferCapacity < frameCount {
                        pointsBufferCapacity = frameCount
                        pointsBuffer = device.makeBuffer(length: byteCount, options: [.storageModeShared])
                    }

                    if let pointsBuffer {
                        let dst = pointsBuffer.contents()
                            .bindMemory(to: SIMD2<Float>.self, capacity: pointsBufferCapacity)
                        let (src, _) = windowSnapshot.getPoints()
                        let scale = params.gain * params.zoom
                        if abs(scale - 1.0) < 0.0001 {
                            memcpy(dst, src, byteCount)
                        } else {
                            for i in 0..<frameCount {
                                dst[i] = src[i] * scale
                            }
                        }
                    }
                    pointCount = frameCount
                }
            }

            // Decay pass
            if let accumTexture,
               let decayPipelineState,
               !needsClear,
               !params.freeze {
                let tau = max(params.tauSeconds, 0.001)
                var decayParams = DecayParams(decay: exp(-dt / tau))
                if let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.setComputePipelineState(decayPipelineState)
                    encoder.setTexture(accumTexture, index: 0)
                    encoder.setBytes(&decayParams, length: MemoryLayout<DecayParams>.stride, index: 0)

                    let w = decayPipelineState.threadExecutionWidth
                    let h = max(1, decayPipelineState.maxTotalThreadsPerThreadgroup / w)
                    let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
                    let threadsPerGrid = MTLSize(width: accumTexture.width, height: accumTexture.height, depth: 1)
                    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                    encoder.endEncoding()
                }
            }

            // Beam deposition pass
            if let accumTexture,
               let beamPipelineState,
               let pointsBuffer,
               pointCount > 1,
               !params.freeze {
                let accumPass = MTLRenderPassDescriptor()
                accumPass.colorAttachments[0].texture = accumTexture
                accumPass.colorAttachments[0].loadAction = needsClear ? .clear : .load
                accumPass.colorAttachments[0].storeAction = .store
                accumPass.colorAttachments[0].clearColor = view.clearColor

                if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: accumPass) {
                    encoder.setRenderPipelineState(beamPipelineState)
                    encoder.setVertexBuffer(pointsBuffer, offset: 0, index: 0)
                    var beamParams = BeamParams(
                        viewportSize: SIMD2<Float>(Float(accumTexture.width), Float(accumTexture.height)),
                        sigma: params.beamSigmaPx,
                        intensity: params.intensity,
                        hue: params.hue,
                        dwellCompensation: params.dwellCompensation ? 1.0 : 0.0
                    )
                    encoder.setVertexBytes(&beamParams, length: MemoryLayout<BeamParams>.stride, index: 1)
                    encoder.setFragmentBytes(&beamParams, length: MemoryLayout<BeamParams>.stride, index: 1)
                    encoder.drawPrimitives(type: .triangleStrip,
                                           vertexStart: 0,
                                           vertexCount: 4,
                                           instanceCount: pointCount - 1)
                    encoder.endEncoding()
                }
            }

            // Bloom pass
            if let accumTexture,
               let bloomTexture,
               let bloomTempTexture,
               let downsamplePipelineState,
               let compositeSampler {
                let downsamplePass = MTLRenderPassDescriptor()
                downsamplePass.colorAttachments[0].texture = bloomTexture
                downsamplePass.colorAttachments[0].loadAction = .clear
                downsamplePass.colorAttachments[0].storeAction = .store
                downsamplePass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

                if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: downsamplePass) {
                    encoder.setRenderPipelineState(downsamplePipelineState)
                    encoder.setFragmentTexture(accumTexture, index: 0)
                    encoder.setFragmentSamplerState(compositeSampler, index: 0)
                    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                    encoder.endEncoding()
                }

                let desiredSigma = max(params.bloomSigmaPx, 0.0)
                if desiredSigma > 0.01, let device = view.device {
                    if abs(desiredSigma - bloomKernelSigma) > 0.01 || bloomKernel == nil {
                        bloomKernel = MPSImageGaussianBlur(device: device, sigma: desiredSigma)
                        bloomKernelSigma = desiredSigma
                    }
                    bloomKernel?.encode(commandBuffer: commandBuffer,
                                        sourceTexture: bloomTexture,
                                        destinationTexture: bloomTempTexture)
                }
            }

            // Composite to screen
            if let compositePipelineState,
               let compositeSampler,
               let accumTexture,
               let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) {
                encoder.setRenderPipelineState(compositePipelineState)
                let bloomTextureForComposite = bloomTempTexture ?? bloomTexture ?? accumTexture
                var compositeParams = CompositeParams(
                    bloomAmount: params.bloomAmount,
                    exposure: 1.1,
                    gamma: 2.2
                )
                encoder.setFragmentBytes(&compositeParams, length: MemoryLayout<CompositeParams>.stride, index: 0)
                encoder.setFragmentTexture(accumTexture, index: 0)
                encoder.setFragmentTexture(bloomTextureForComposite, index: 1)
                encoder.setFragmentSamplerState(compositeSampler, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                encoder.endEncoding()
            }

            commandBuffer.present(drawable)
            commandBuffer.commit()
            needsClear = false
        }

        private func ensureTextures(view: MTKView) {
            guard let device = view.device else { return }
            let width = max(1, Int(view.drawableSize.width.rounded()))
            let height = max(1, Int(view.drawableSize.height.rounded()))

            if accumTexture?.width != width || accumTexture?.height != height {
                let accumDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float,
                    width: width,
                    height: height,
                    mipmapped: false
                )
                accumDescriptor.usage = [.renderTarget, .shaderRead, .shaderWrite]
                accumDescriptor.storageMode = .private
                accumTexture = device.makeTexture(descriptor: accumDescriptor)

                let bloomWidth = max(1, width / 4)
                let bloomHeight = max(1, height / 4)
                let bloomDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float,
                    width: bloomWidth,
                    height: bloomHeight,
                    mipmapped: false
                )
                bloomDescriptor.usage = [.renderTarget, .shaderRead, .shaderWrite]
                bloomDescriptor.storageMode = .private
                bloomTexture = device.makeTexture(descriptor: bloomDescriptor)
                bloomTempTexture = device.makeTexture(descriptor: bloomDescriptor)
                needsClear = true
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(paramsStore: paramsStore,
                    sampleStore: sampleStore,
                    sampleRate: sampleRate)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.contentScaleFactor = UIScreen.main.nativeScale
        context.coordinator.configure(view: view)
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.updateSampleRate(sampleRate)
    }
}
