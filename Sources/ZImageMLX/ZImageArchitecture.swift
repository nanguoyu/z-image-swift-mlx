import Foundation
import CoreGraphics
import MLX
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
        // TODO(phase0): Qwen3-4B encode → embeddings. Loaded here, released by the engine after.
        throw ZImageError.notImplemented("encode (Qwen3-4B text encoder)")
    }

    public func makeDenoiser(source: WeightSource) throws -> any Denoiser {
        // TODO(phase0): build the S3-DiT denoiser (patch-embed → single-stream blocks → unembed).
        throw ZImageError.notImplemented("makeDenoiser (S3-DiT single-stream)")
    }

    public func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                              source: WeightSource) throws -> MLXArray {
        // TODO(phase0): seeded latent (+ optional img2img encode of `reference`).
        throw ZImageError.notImplemented("initialLatent")
    }

    public func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage {
        // TODO(phase0): VAE decode → CGImage.
        throw ZImageError.notImplemented("decode (VAE)")
    }
}

enum ZImageError: Error, CustomStringConvertible {
    case notImplemented(String)
    var description: String {
        switch self { case .notImplemented(let what): return "ZImage: not implemented — \(what)" }
    }
}
