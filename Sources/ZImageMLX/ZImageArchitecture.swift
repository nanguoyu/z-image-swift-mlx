import Foundation
import CoreGraphics
@preconcurrency import MLX
import MLXRandom
import Tokenizers
import DiffusionCore

/// Z-Image (Tongyi) — single-stream S3-DiT with a Qwen3-4B text encoder.
///
/// Conforms to the core `DiffusionArchitecture` seam so the shared `MLXDiffusionEngine` can
/// drive it, including block-streaming partial load. Scaffold: the seam is declared; the
/// implementation lands in Phase 0 (this is the first non-FLUX model, used to measure the
/// real per-architecture Swift cost called out in the blueprint).
public struct ZImageArchitecture: DiffusionArchitecture {

    public init() {}

    public static let spec = ArchitectureSpec(
        family: .zImage,
        latentChannels: 16,
        defaultSampler: .flowMatchEuler,
        defaultSteps: 8,
        defaultGuidance: 1.0)

    public func encode(_ prompt: String, negative: String?, source: WeightSource) async throws -> Conditioning {
        // TODO(phase0): resolve the tokenizer + load Qwen3-4B weights from `source`'s on-disk
        // text_encoder/ folder instead of the hub id; pass `enable_thinking` via the template;
        // take the second-to-last hidden state (already done in the encoder).
        let tokenizer = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3-4B")
        let messages: [Message] = [["role": "user", "content": prompt]]
        let ids = try tokenizer.applyChatTemplate(messages: messages)
        let trimmed = Array(ids.prefix(ZImageConfig.TextEncoder.maxSequenceLength))
        let tokens = MLXArray(trimmed.map { Int32($0) }).reshaped([1, trimmed.count])
        let hidden = Qwen3TextEncoder().hiddenStates(tokens)   // [1, N, 2560]
        return Conditioning(embeddings: hidden)
    }

    public func makeDenoiser(source: WeightSource) throws -> any Denoiser {
        // Builds the S3-DiT denoiser structure. TODO(phase0): load the 4-bit weights from
        // `source` into the module tree (key map in IMPLEMENTATION.md) before running.
        ZImageDenoiser()
    }

    public func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                              source: WeightSource) throws -> MLXArray {
        // Seeded Gaussian latent in VAE space [1, C, H/8, W/8].
        // TODO(phase0): img2img — encode `reference` through the VAE and blend by `strength`.
        let f = ZImageConfig.VAE.downsampleFactor
        let c = ZImageConfig.VAE.latentChannels
        return MLXRandom.normal([1, c, size.height / f, size.width / f], key: MLXRandom.key(seed))
    }

    public func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage {
        // TODO(phase0): load the VAE weights from `source` and cache the instance.
        let vae = ZImageVAE()
        let imageNHWC = vae.decode(latent)              // [B, H, W, 3] in [-1, 1]
        guard let image = ImageConversion.cgImage(fromHWC: imageNHWC[0], range: .signed) else {
            throw ZImageError.notImplemented("VAE decode produced no image")
        }
        return image
    }
}

enum ZImageError: Error, CustomStringConvertible {
    case notImplemented(String)
    var description: String {
        switch self { case .notImplemented(let what): return "ZImage: not implemented — \(what)" }
    }
}
