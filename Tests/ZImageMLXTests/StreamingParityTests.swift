import XCTest
@preconcurrency import MLX
import MLXNN
import MLXRandom
import DiffusionCore
import Tokenizers
@testable import ZImageMLX

/// Step 7 — the CORRECTNESS GATE for the iPhone block-streaming path.
///
/// The streaming path runs the SAME `ZImageTransformerBlock` math as the verified resident macOS
/// path (`ZImagePipeline` -> `ZImageDenoiser()`), differing ONLY in how the 30 main `layers.*`
/// blocks are loaded: resident loads the whole transformer tree once; streaming loads/frees each
/// block per step from a `RangedFileWeightSource`. So correctness reduces to: for an identical
/// seed/prompt/steps/size, the STREAMING denoise output must MATCH the RESIDENT one.
///
/// This test, GIVEN the real checkpoint dir in env var `ZIMAGE_CHECKPOINT`, runs the full denoise
/// loop TWICE over one shared conditioning (encoded once from the real text encoder — sharing it
/// isolates transformer-load correctness and avoids paying the ~Qwen3-4B encode twice), then
/// asserts the two final latents are pixel-close (allClose) at a high PSNR. It also decodes BOTH
/// latents through the same VAE and asserts the resulting images match — the end-to-end image
/// parity the milestone calls for.
///
/// Config is deliberately SMALL (512x512, 2 steps, fixed seed) so the gate runs in minutes once the
/// weights are loaded. If `ZIMAGE_CHECKPOINT` is unset the test SKIPS (so CI without the checkpoint
/// stays green); the runner sets it.
final class StreamingParityTests: XCTestCase {

    /// PSNR (dB) between two tensors, computed in fp32. Higher = closer; ∞ for identical.
    private func psnr(_ a: MLXArray, _ b: MLXArray) -> Double {
        let af = a.asType(.float32), bf = b.asType(.float32)
        let mse = (af - bf).square().mean().item(Float.self)
        if mse == 0 { return .infinity }
        // The latent/image dynamic range here is O(1)..O(10); use the observed peak abs value as the
        // signal peak so the dB figure is meaningful rather than assuming a fixed [0,1] range.
        let peak = max(af.abs().max().item(Float.self), bf.abs().max().item(Float.self), 1e-6)
        return 20.0 * Foundation.log10(Double(peak)) - 10.0 * Foundation.log10(Double(mse))
    }

    /// Run the shared Z-Image denoise loop. `streaming == false` uses the pre-built resident denoiser
    /// (all `layers.*` resident); `true` drives the engine's per-block load -> run -> materialize ->
    /// release -> clearCache lifecycle against `txSource`. Both share conditioning/seed/steps/size.
    private func denoiseLatent(denoiser: ZImageDenoiser, streaming: Bool, txSource: WeightSource?,
                               conditioning: Conditioning, size: Int, steps: Int, seed: UInt64) throws -> MLXArray {
        let factor = ZImageConfig.VAE.downsampleFactor
        let channels = ZImageConfig.VAE.latentChannels
        let sampler = FlowMatchEulerSampler(shift: ZImageConfig.Scheduler.shift)
        let sigmas = sampler.timesteps(steps: steps)
        var latent = MLXRandom.normal([1, channels, size / factor, size / factor],
                                      key: MLXRandom.key(seed)).asType(.bfloat16)
        for i in 0..<steps {
            let t = sigmas[i], tNext = sigmas[i + 1]
            let timestep = MLXArray(t)
            var hidden = denoiser.embed(latent: latent, timestep: timestep, conditioning: conditioning)
            for block in denoiser.blocks {
                if streaming { try block.load(from: txSource!) }
                hidden = block(hidden, conditioning: conditioning, timestep: timestep)
                if streaming {
                    MLX.eval(hidden)
                    block.release()
                    MLX.GPU.clearCache()
                }
            }
            latent = sampler.step(latent: latent, modelOutput: denoiser.unembed(hidden), t: t, tPrev: tNext)
            MLX.eval(latent)
        }
        return latent
    }

    func testStreamingMatchesResident() async throws {
        // xcodebuild forwards host env vars into the test runner only when prefixed `TEST_RUNNER_`,
        // so accept either the bare name (e.g. `swift test`) or the forwarded one.
        let env = ProcessInfo.processInfo.environment
        guard let ckpt = (env["ZIMAGE_CHECKPOINT"] ?? env["TEST_RUNNER_ZIMAGE_CHECKPOINT"]), !ckpt.isEmpty else {
            throw XCTSkip("set ZIMAGE_CHECKPOINT to the real Z-Image model dir to run the parity gate")
        }
        let modelDir = URL(fileURLWithPath: ckpt, isDirectory: true)
        let transformerDir = modelDir.appendingPathComponent("transformer")
        XCTAssertTrue(FileManager.default.fileExists(atPath: transformerDir.path),
                      "ZIMAGE_CHECKPOINT must contain a transformer/ folder")

        let prompt = "a red apple on a wooden table"
        let size = 512, steps = 2
        let seed: UInt64 = 1234

        // --- 1. Encode the prompt ONCE via the real text encoder; share the conditioning. ---
        // (Identical conditioning into both denoise loops isolates the variable under test — how the
        // transformer blocks are loaded — from any text-encoder/tokenizer nondeterminism.)
        let arch = ZImageArchitecture()
        let encStreaming = try ZImageComponentSource.open(modelDirectory: modelDir, streaming: true)
        let conditioning = try await arch.encode(prompt, negative: nil, source: encStreaming)
        MLX.eval(conditioning.embeddings)
        arch.releaseTextEncoder()
        MLX.GPU.clearCache()

        // --- 2. RESIDENT path: whole-tree load of the transformer (the verified ZImagePipeline load). ---
        let resident = ZImageDenoiser(streaming: false)
        ZImageWeights.load(try ZImageWeights.tensors(in: transformerDir), into: resident)
        MLX.eval(resident)
        let residentLatent = try denoiseLatent(denoiser: resident, streaming: false, txSource: nil,
                                               conditioning: conditioning, size: size, steps: steps, seed: seed)
        MLX.eval(residentLatent)

        // --- 3. STREAMING path: shared submodules resident, layers.* streamed per block per step. ---
        // Build exactly as ZImageArchitecture.makeDenoiser does: loadShared(skipPrefixes:["layers."]).
        let streamSource = try ZImageComponentSource.open(modelDirectory: modelDir, streaming: true)
        guard let txSource = streamSource.subSource(.transformer) else {
            return XCTFail("composite source has no transformer sub-source")
        }
        let streaming = ZImageDenoiser(streaming: true)
        try ZImageWeights.loadShared(from: txSource, into: streaming, skipPrefixes: ["layers."])
        let streamingLatent = try denoiseLatent(denoiser: streaming, streaming: true, txSource: txSource,
                                                conditioning: conditioning, size: size, steps: steps, seed: seed)
        MLX.eval(streamingLatent)

        // --- 4. LATENT parity: the streamed output must match the resident output. ---
        XCTAssertEqual(residentLatent.shape, streamingLatent.shape, "latent shape mismatch")
        let latentPSNR = psnr(residentLatent, streamingLatent)
        let close = allClose(residentLatent.asType(.float32), streamingLatent.asType(.float32),
                             rtol: 1e-3, atol: 1e-2).item(Bool.self)
        print("[parity] latent PSNR = \(latentPSNR) dB, allClose(rtol=1e-3,atol=1e-2) = \(close)")
        XCTAssertGreaterThan(latentPSNR, 50.0,
                             "streaming latent diverged from resident (PSNR \(latentPSNR) dB) — block load/key-alignment is wrong")
        XCTAssertTrue(close, "streaming latent not allClose to resident (PSNR \(latentPSNR) dB)")

        // --- 5. IMAGE parity (end-to-end): decode BOTH latents through the same VAE. ---
        let vaeSource = try ZImageComponentSource.open(modelDirectory: modelDir, streaming: false)
        let residentImage = try await arch.decode(residentLatent, source: vaeSource)
        let streamingImage = try await arch.decode(streamingLatent, source: vaeSource)
        XCTAssertEqual(residentImage.width, streamingImage.width)
        XCTAssertEqual(residentImage.height, streamingImage.height)
        let (rPix, sPix) = (Self.pixels(residentImage), Self.pixels(streamingImage))
        let imgPSNR = psnr(MLXArray(rPix), MLXArray(sPix))
        print("[parity] image PSNR = \(imgPSNR) dB over \(rPix.count) bytes")
        XCTAssertGreaterThan(imgPSNR, 40.0, "decoded images diverged (PSNR \(imgPSNR) dB)")
    }

    /// Flatten a CGImage to RGBA8 bytes for a numeric comparison.
    private static func pixels(_ image: CGImage) -> [Float] {
        let w = image.width, h = image.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buf.map { Float($0) }
    }
}
