import Foundation
import CoreGraphics
@preconcurrency import MLX
import MLXRandom
import Tokenizers
import DiffusionCore

/// Z-Image (Tongyi) — single-stream S3-DiT with a Qwen3-4B text encoder.
///
/// Conforms to the core `DiffusionArchitecture` seam so the shared `MLXDiffusionEngine` can drive it,
/// including block-streaming partial load (the iPhone path). The resident macOS path stays on
/// `ZImageFacadeEngine` + `ZImagePipeline` and is untouched.
///
/// NAMESPACE SEAM (R5): the engine hands ONE flat `WeightSource`, but Z-Image is three component
/// trees (`text_encoder/`, `transformer/`, `vae/`) whose key spaces collide. So this architecture
/// expects that single source to actually be a `ZImageComponentSource` — a composite that owns one
/// sub-`WeightSource` per component folder — and pulls the per-component sub-source it needs in each
/// phase (`encode` → text_encoder, `makeDenoiser` → transformer, `decode` → vae). The app builds the
/// composite via `ZImageComponentSource.open(modelDirectory:streaming:)` and hands it to the engine's
/// `load(...)`.
///
/// State (resident text encoder, tokenizer, VAE) is held by reference and guarded by a lock so the
/// `Sendable` architecture is safe to share across the engine actor's awaits.
public final class ZImageArchitecture: DiffusionArchitecture, @unchecked Sendable {

    private let lock = NSLock()
    // Held only between `encode` and `releaseTextEncoder` (two-phase staging: the encoder and the
    // streaming transformer never co-reside).
    private var textEncoder: Qwen3TextEncoder?
    private var tokenizer: Tokenizer?
    // VAE is built+bound lazily in `decode` and cached for reuse across generations.
    private var vae: ZImageVAE?

    public init() {}

    public static let spec = ArchitectureSpec(
        family: .zImage,
        latentChannels: 16,
        defaultSampler: .flowMatchEuler,
        defaultSteps: 8,
        defaultGuidance: 1.0,
        samplerShift: ZImageConfig.Scheduler.shift,
        samplerShiftTerminal: ZImageConfig.Scheduler.shiftTerminal)

    public enum ArchitectureError: Error, CustomStringConvertible {
        case notComponentSource
        case missingComponent(String)
        public var description: String {
            switch self {
            case .notComponentSource:
                return "ZImageArchitecture: expected a ZImageComponentSource (per-component text_encoder/transformer/vae). Build one with ZImageComponentSource.open(modelDirectory:streaming:) before MLXDiffusionEngine.load(...)."
            case .missingComponent(let c):
                return "ZImageArchitecture: the ZImageComponentSource has no '\(c)' sub-source"
            }
        }
    }

    /// Resolve the composite source into one component's sub-source (R5 collision resolution).
    private func component(_ c: ZImageComponentSource.Component, in source: WeightSource) throws -> any WeightSource {
        guard let composite = source as? ZImageComponentSource else { throw ArchitectureError.notComponentSource }
        guard let sub = composite.subSource(c) else { throw ArchitectureError.missingComponent(c.rawValue) }
        return sub
    }

    public func encode(_ prompt: String, negative: String?, source: WeightSource) async throws -> Conditioning {
        guard let composite = source as? ZImageComponentSource else { throw ArchitectureError.notComponentSource }
        let encSource = try component(.textEncoder, in: source)

        // Resolve the tokenizer from the model's own `tokenizer/` folder (no hub round-trip on device);
        // fall back to the published hub id if the folder wasn't shipped.
        let tok: Tokenizer
        if let dir = composite.tokenizerDirectory {
            tok = try await AutoTokenizer.from(modelFolder: dir)
        } else {
            tok = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3-4B")
        }

        // Build + load the Qwen3-4B encoder from the text_encoder component, hold it resident until
        // releaseTextEncoder(). The whole text-encoder tree is loaded in one pass (it is small enough
        // to stage transiently and is freed before the transformer streams).
        let encoder = Qwen3TextEncoder()
        try ZImageWeights.loadShared(from: encSource, into: encoder, skipPrefixes: [])

        let messages: [Message] = [["role": "user", "content": prompt]]
        // Match mflux's call (chat_template_kwargs={'enable_thinking': True}). The Qwen3 template only
        // branches when enable_thinking is explicitly false, so this is a no-op vs the default, but it
        // pins the prompt tokens to mflux's regardless of any tokenizer default.
        let ids = try tok.applyChatTemplate(messages: messages, tools: nil,
                                            additionalContext: ["enable_thinking": true])
        let length = min(ids.count, ZImageConfig.TextEncoder.maxSequenceLength)
        let tokens = MLXArray(ids.prefix(length).map { Int32($0) }).reshaped([1, length])
        let hidden = encoder.hiddenStates(tokens)   // [1, N, 2560]
        MLX.eval(hidden)                             // materialize before we drop the encoder

        // Free the text-encoder source's eagerly-held arrays now; the module is dropped in
        // releaseTextEncoder(), and only once BOTH refs are gone does the ~2 GB actually free.
        composite.releaseComponent(.textEncoder)
        lock.lock(); self.textEncoder = encoder; self.tokenizer = tok; lock.unlock()
        return Conditioning(embeddings: hidden)
    }

    public func releaseTextEncoder() {
        lock.lock(); textEncoder = nil; tokenizer = nil; lock.unlock()
        MLX.GPU.clearCache()
    }

    public func releaseCachedResources() {
        lock.lock()
        textEncoder = nil
        tokenizer = nil
        vae = nil
        lock.unlock()
        MLX.GPU.clearCache()
    }

    public func makeDenoiser(source: WeightSource) throws -> any Denoiser {
        // Streaming S3-DiT: the 30 main `layers.*` blocks are NOT resident; each ZImageStreamableBlock
        // loads/frees its own block per step from the transformer source (the engine drives
        // load → run → eval → release → clearCache). The small shared submodules (embedders, refiners,
        // pad tokens, final layer) stay resident and are filled here, once, from the transformer
        // component — skipping `layers.` so the per-block streaming owns those keys.
        let txSource = try component(.transformer, in: source)
        let denoiser = ZImageDenoiser(streaming: true)
        try ZImageWeights.loadShared(from: txSource, into: denoiser, skipPrefixes: ["layers."])
        return denoiser
    }

    public func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                              source: WeightSource) throws -> MLXArray {
        // Seeded Gaussian latent in VAE space [1, C, H/8, W/8], bf16 to match the transformer math.
        // TODO(phase0): img2img — encode `reference` through the VAE and blend by `strength`.
        let f = ZImageConfig.VAE.downsampleFactor
        let c = ZImageConfig.VAE.latentChannels
        return MLXRandom.normal([1, c, size.height / f, size.width / f], key: MLXRandom.key(seed))
            .asType(.bfloat16)
    }

    public func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage {
        let vaeSource = try component(.vae, in: source)
        // Build + bind the VAE once, then cache it for subsequent generations.
        let vae: ZImageVAE
        lock.lock(); let cached = self.vae; lock.unlock()
        if let cached {
            vae = cached
        } else {
            let v = ZImageVAE()
            try ZImageWeights.loadShared(from: vaeSource, into: v, skipPrefixes: [])
            lock.lock(); self.vae = v; lock.unlock()
            vae = v
        }
        let imageNHWC = vae.decode(latent).asType(.float32)
        MLX.eval(imageNHWC)
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
