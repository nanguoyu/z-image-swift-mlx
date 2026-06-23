@preconcurrency import MLX
import MLXNN
import DiffusionCore
import Foundation

// Z-Image S3-DiT denoiser assembly: patch-embed the latent, build the single-stream sequence
// (caption tokens + image tokens), run the AdaLN-modulated blocks, then unembed back to latent
// space. Conforms to swift-diffusion-core's `Denoiser` so MLXDiffusionEngine can drive it.
//
// NOTE: structure + weight keys follow the reference (IMPLEMENTATION.md). The forward numerics
// — the 3D-axes RoPE (1D placeholder for now), AdaLN application, and patch packing — need GPU
// parity validation against the Python reference. Weight loading (4-bit) is a TODO.

/// Timestep → sinusoidal embedding → MLP → `dim`. Keys: `t_embedder.mlp.{0,1}`.
final class TimestepEmbedder: Module {
    let mlp: [Linear]
    private let freqDim = 256
    init(dim: Int) {
        self.mlp = [Linear(256, dim), Linear(dim, dim)]
        super.init()
    }
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        mlp[1](silu(mlp[0](TimestepEmbedder.sinusoidal(t, dim: freqDim))))
    }
    static func sinusoidal(_ t: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let scale = -Foundation.log(10000.0) / Double(half)
        let idx = MLXArray((0..<half).map { Float(Double($0) * scale) })
        let freqs = exp(idx)                              // [half]
        let args = t.reshaped([-1, 1]) * freqs.reshaped([1, -1])   // [B, half]
        return concatenated([cos(args), sin(args)], axis: -1)      // [B, dim]
    }
}

/// Caption projection: RMSNorm + Linear (2560 → dim). Keys: `cap_embedder.{0,1}`.
final class CaptionEmbedder: Module {
    @ModuleInfo(key: "0") var norm: RMSNorm
    @ModuleInfo(key: "1") var proj: Linear
    init(inDim: Int, dim: Int) {
        self._norm.wrappedValue = RMSNorm(dimensions: inDim, eps: ZImageConfig.DiT.rmsEps)
        self._proj.wrappedValue = Linear(inDim, dim, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(norm(x)) }
}

/// Adapts a `ZImageTransformerBlock` to the engine's `StreamableBlock`. The block derives its
/// AdaLN modulation from the timestep via the shared `TimestepEmbedder`. `load`/`release` are
/// no-ops here (weights are resident); the iPhone streaming path will load per-block ranges.
final class ZImageStreamableBlock: StreamableBlock {
    let index: Int
    let approximateBytes: Int64
    private let block: ZImageTransformerBlock
    private let timeEmbedder: TimestepEmbedder

    init(index: Int, block: ZImageTransformerBlock, timeEmbedder: TimestepEmbedder, approximateBytes: Int64) {
        self.index = index
        self.block = block
        self.timeEmbedder = timeEmbedder
        self.approximateBytes = approximateBytes
    }
    func load(from source: WeightSource) throws {}
    func callAsFunction(_ x: MLXArray, conditioning: Conditioning, timestep: MLXArray) -> MLXArray {
        block(x, timeEmb: timeEmbedder(timestep))
    }
    func release() {}
}

public final class ZImageDenoiser: Module, Denoiser {
    public let blocks: [any StreamableBlock]
    private let transformerBlocks: [ZImageTransformerBlock]
    private let timeEmbedder: TimestepEmbedder
    private let capEmbedder: CaptionEmbedder
    private let xEmbedder: Linear            // patched latent (16) -> dim
    private let finalNorm: RMSNorm
    private let finalProj: Linear            // dim -> patched latent (16)

    // Set during `embed`, used by `unembed` (same generation).
    private var hp = 0, wp = 0, captionLength = 0

    public override init() {
        let dim = ZImageConfig.DiT.dim
        let te = TimestepEmbedder(dim: dim)
        self.timeEmbedder = te
        self.capEmbedder = CaptionEmbedder(inDim: ZImageConfig.DiT.captionDim, dim: dim)
        self.xEmbedder = Linear(ZImageConfig.DiT.patchedChannels, dim, bias: true)
        self.finalNorm = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self.finalProj = Linear(dim, ZImageConfig.DiT.patchedChannels, bias: true)
        let tblocks = (0..<ZImageConfig.DiT.layers).map { _ in ZImageTransformerBlock(dim: dim) }
        self.transformerBlocks = tblocks
        self.blocks = tblocks.enumerated().map { i, b in
            ZImageStreamableBlock(index: i, block: b, timeEmbedder: te, approximateBytes: 120_000_000)
        }
        super.init()
    }

    /// Patchify the latent, project, and build the single-stream sequence [caption ; image].
    public func embed(latent: MLXArray, timestep: MLXArray, conditioning: Conditioning) -> MLXArray {
        let b = latent.dim(0), c = latent.dim(1), h = latent.dim(2), w = latent.dim(3)
        let p = ZImageConfig.DiT.patchSize
        hp = h / p; wp = w / p
        // [B,C,H,W] -> [B, hp, wp, C*p*p]
        let patches = latent
            .reshaped([b, c, hp, p, wp, p])
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped([b, hp * wp, c * p * p])
        let imageTokens = xEmbedder(patches)                       // [B, N, dim]
        let captionTokens = capEmbedder(conditioning.embeddings)   // [B, L, dim]
        captionLength = captionTokens.dim(1)
        return concatenated([captionTokens, imageTokens], axis: 1) // [B, L+N, dim]
    }

    /// Drop the caption tokens, project the image tokens back to patched latent, and unpatchify.
    public func unembed(_ hidden: MLXArray) -> MLXArray {
        let imageTokens = split(hidden, indices: [captionLength], axis: 1)[1]   // [B, N, dim]
        let patches = finalProj(finalNorm(imageTokens))                          // [B, N, C*p*p]
        let p = ZImageConfig.DiT.patchSize
        let c = ZImageConfig.DiT.vaeLatentChannels
        let b = patches.dim(0)
        // [B, N, C*p*p] -> [B, C, H, W]
        return patches
            .reshaped([b, hp, wp, c, p, p])
            .transposed(0, 3, 1, 4, 2, 5)
            .reshaped([b, c, hp * p, wp * p])
    }
}
