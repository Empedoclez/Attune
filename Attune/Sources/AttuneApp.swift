import SwiftUI

@main
struct AttuneApp: App {
    @StateObject private var viewModel = ScopeViewModel()

    var body: some Scene {
        WindowGroup {
            ScopeScreen(viewModel: viewModel)
                .preferredColorScheme(.dark)
                .statusBarHidden(true)
        }
    }
}

// MARK: - View Model

@MainActor
final class ScopeViewModel: ObservableObject {
    let paramsStore = ScopeParamsStore()
    let sampleStore = SampleStore(capacityFrames: 65536) // ~1.4s at 48kHz
    private var audioCapture: AudioCapture?

    @Published var isRunning = false
    @Published var errorMessage: String?
    @Published var sampleRate: Double = 48000.0
    @Published var deviceName: String = ""

    init() {
        // Set defaults tuned for iPhone voice at arm's length
        paramsStore.modify { params in
            params.gain = 3.0          // Boost — iPhone mics are quieter than interfaces
            params.zoom = 1.0
            params.beamSigmaPx = 1.5   // Slightly thicker for small screen
            params.tauSeconds = 0.05   // Very short persistence — crisp mode
            params.intensity = 1.2
            params.bloomAmount = 0.2   // Subtle bloom
            params.bloomSigmaPx = 6.0
            params.hue = 0.35         // Green phosphor
            params.freeze = false
            params.dwellCompensation = false
        }
    }

    func startAudio() {
        guard !isRunning else { return }

        let capture = AudioCapture(sampleStore: sampleStore)
        self.audioCapture = capture

        Task {
            do {
                try await capture.requestPermissionAndStart()
                await MainActor.run {
                    self.isRunning = true
                    self.errorMessage = nil
                    if let info = capture.deviceInfo {
                        self.sampleRate = info.sampleRate
                        self.deviceName = info.name
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isRunning = false
                }
            }
        }
    }

    func stopAudio() {
        audioCapture?.stop()
        audioCapture = nil
        isRunning = false
    }
}

// MARK: - Main Screen

struct ScopeScreen: View {
    @ObservedObject var viewModel: ScopeViewModel

    var body: some View {
        ZStack {
            // The scope fills the entire screen
            MetalScopeView(
                paramsStore: viewModel.paramsStore,
                sampleStore: viewModel.sampleStore,
                sampleRate: viewModel.sampleRate
            )
            .ignoresSafeArea()

            // Overlay UI
            VStack {
                Spacer()

                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding(8)
                        .background(.black.opacity(0.7))
                        .cornerRadius(8)
                        .padding(.bottom, 8)
                }

                if !viewModel.isRunning {
                    Button(action: {
                        viewModel.startAudio()
                    }) {
                        Text("Start")
                            .font(.title3)
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                            .padding(.horizontal, 32)
                            .padding(.vertical, 12)
                            .background(.white.opacity(0.15))
                            .cornerRadius(24)
                    }
                    .padding(.bottom, 48)
                } else {
                    // Minimal gain slider at the bottom
                    HStack(spacing: 16) {
                        Text("Gain")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.5))

                        Slider(
                            value: Binding(
                                get: { Double(viewModel.paramsStore.gain) },
                                set: { viewModel.paramsStore.gain = Float($0) }
                            ),
                            in: 0.5...15.0
                        )
                        .tint(.white.opacity(0.4))
                    }
                    .padding(.horizontal, 32)
                    .padding(.vertical, 12)
                    .background(.black.opacity(0.4))
                    .cornerRadius(16)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 32)
                }
            }
        }
        .onAppear {
            viewModel.startAudio()
        }
    }
}
