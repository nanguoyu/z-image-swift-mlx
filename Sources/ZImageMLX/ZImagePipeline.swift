@preconcurrency import MLX
import MLXNN
import MLXRandom
import DiffusionCore
import Tokenizers
import CoreGraphics
import Foundation

/// High-level Z-Image text-to-image pipeline: loads the three component trees (Qwen3-4B text
/// encoder, S3-DiT transformer, AutoencoderKL VAE) + the tokenizer from a model directory, then
/// runs encode → flow-match denoise → VAE decode. This is the convenience/loading layer (analogous
/// to FLUX's `Flux2Pipeline`); the same `ZImageDenoiser` also conforms to `Denoiser` for the
/// block-streaming engine path used on iPhone.
///
/// `modelDirectory` is a folder laid out like `deepsweet/Z-Image-Turbo-6B-MLX-Q4`:
/// `text_encoder/`, `transformer/`, `vae/`, `tokenizer/` (each `*.safetensors` + index.json).
public final class ZImagePipeline: @unchecked Sendable {
    public enum PipelineError: Error, CustomStringConvertible {
        case notLoaded
        case decodeFailed
        case invalidSize(Int)
        public var description: String {
            switch self {
            case .notLoaded: return "ZImagePipeline: call loadModels() before generate()"
            case .decodeFailed: return "ZImagePipeline: VAE decode produced no image"
            case .invalidSize(let s): return "ZImagePipeline: size \(s) must be a multiple of 16"
            }
        }
    }

    private let modelDirectory: URL
    private var encoder: Qwen3TextEncoder?
    private var denoiser: ZImageDenoiser?
    private var vae: ZImageVAE?
    private var tokenizer: Tokenizer?

    public init(modelDirectory: URL) { self.modelDirectory = modelDirectory }

    /// Load + quantize all three trees and the tokenizer. `progress` reports 0…1 across components.
    public func loadModels(progress: (@Sendable (Double) -> Void)? = nil) async throws {
        func load(_ component: String, _ module: Module) throws {
            let dir = modelDirectory.appendingPathComponent(component)
            ZImageWeights.load(try ZImageWeights.tensors(in: dir), into: module)
        }
        let enc = Qwen3TextEncoder(); try load("text_encoder", enc); progress?(0.4)
        let dn = ZImageDenoiser(); try load("transformer", dn); progress?(0.85)
        let v = ZImageVAE(); try load("vae", v); progress?(0.95)
        let tok = try await AutoTokenizer.from(modelFolder: modelDirectory.appendingPathComponent("tokenizer"))
        self.encoder = enc; self.denoiser = dn; self.vae = v; self.tokenizer = tok
        progress?(1.0)
    }

    public func unload() { encoder = nil; denoiser = nil; vae = nil; tokenizer = nil; MLX.GPU.clearCache() }

    /// Encode the prompt to Qwen3-4B's layer[-2] hidden state (the value Z-Image conditions on).
    public func encode(_ prompt: String) throws -> Conditioning {
        guard let encoder, let tokenizer else { throw PipelineError.notLoaded }
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let ids = try tokenizer.applyChatTemplate(messages: messages)
        let length = min(ids.count, ZImageConfig.TextEncoder.maxSequenceLength)
        let tokens = MLXArray(ids.prefix(length).map { Int32($0) }).reshaped([1, length])
        return Conditioning(embeddings: encoder.hiddenStates(tokens))
    }

    /// Text-to-image. `size` is the square side in pixels (multiple of 16; 1024 is native).
    public func generate(prompt: String, size: Int = 1024, steps: Int = ZImageConfig.Scheduler.defaultSteps,
                         seed: UInt64 = 0, progress: (@Sendable (Int, Int) -> Void)? = nil) throws -> CGImage {
        guard let denoiser, let vae else { throw PipelineError.notLoaded }
        // VAE downsamples by 8 and the DiT patchifies by 2, so the latent side must be even:
        // require size to be a multiple of 16 (fail at the call site, not deep in the transformer).
        let factor = ZImageConfig.VAE.downsampleFactor
        guard size % (factor * ZImageConfig.DiT.patchSize) == 0 else { throw PipelineError.invalidSize(size) }
        let conditioning = try encode(prompt)
        let sampler = FlowMatchEulerSampler(shift: ZImageConfig.Scheduler.shift,
                                            shiftTerminal: ZImageConfig.Scheduler.shiftTerminal)
        let sigmas = sampler.timesteps(steps: steps)
        let channels = ZImageConfig.VAE.latentChannels
        var latent = MLXRandom.normal([1, channels, size / factor, size / factor], key: MLXRandom.key(seed)).asType(.bfloat16)
        for i in 0..<steps {
            let t = sigmas[i], tNext = sigmas[i + 1]
            let timestep = MLXArray(t)
            var hidden = denoiser.embed(latent: latent, timestep: timestep, conditioning: conditioning)
            for block in denoiser.blocks { hidden = block(hidden, conditioning: conditioning, timestep: timestep) }
            latent = sampler.step(latent: latent, modelOutput: denoiser.unembed(hidden), t: t, tPrev: tNext)
            eval(latent)
            progress?(i + 1, steps)
        }
        let imageNHWC = vae.decode(latent).asType(.float32)
        eval(imageNHWC)
        guard let image = ImageConversion.cgImage(fromHWC: imageNHWC[0], range: .signed) else {
            throw PipelineError.decodeFailed
        }
        return image
    }
}
