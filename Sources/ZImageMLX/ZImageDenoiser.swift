@preconcurrency import MLX
import MLXNN
import DiffusionCore
import Foundation

// Z-Image S3-DiT denoiser assembly: patch-embed the latent, refine the image and caption streams
// separately (noise_refiner / context_refiner), build the single-stream sequence [caption ; image],
// run the AdaLN-modulated main `layers`, then unembed back to latent space. Conforms to
// swift-diffusion-core's `Denoiser` so MLXDiffusionEngine can drive the main layers as streamable
// blocks.
//
// Module structure + weight keys follow the reference checkpoint (IMPLEMENTATION.md): `layers.N`,
// `noise_refiner.N`, `context_refiner.N`, `t_embedder.mlp.{0,1}`, `cap_embedder.{0,1}`,
// `all_x_embedder.2-1`, `all_final_layer.2-1.{norm_final,linear,adaLN_modulation.0}`,
// `x_pad_token`, `cap_pad_token`. The forward numerics — 3D-axes RoPE (1D placeholder), AdaLN
// application, the refiner flow, and pad-token handling — still need GPU parity validation.

/// Timestep → sinusoidal embedding → MLP → `dim`. Keys: `t_embedder.linear1`, `t_embedder.linear2`.
final class TimestepEmbedder: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    private let freqDim = 256
    init(dim: Int) {
        self._linear1.wrappedValue = Linear(256, dim)
        self._linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        // Flow-match timestep t∈[0,1] is scaled by t_scale (1000) before the sinusoidal embedding.
        linear2(silu(linear1(TimestepEmbedder.sinusoidal(t * ZImageConfig.DiT.tScale, dim: freqDim))))
    }
    static func sinusoidal(_ t: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let scale = -Foundation.log(10000.0) / Double(half)
        let idx = MLXArray((0..<half).map { Float(Double($0) * scale) })
        let freqs = exp(idx)                                       // [half]
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

/// Patch embedder container. The reference keys the single patch-size variant `2-1`, so the Linear
/// lives at `all_x_embedder.2-1`.
final class ZImageXEmbedder: Module {
    @ModuleInfo(key: "2-1") var embed: Linear
    init(dim: Int) {
        self._embed.wrappedValue = Linear(ZImageConfig.DiT.patchedChannels, dim, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { embed(x) }
}

/// Final layer: AdaLN-modulated norm + projection back to patched-latent channels. The reference
/// keys only `linear` and `adaLN_modulation.0` — `norm_final` carries NO learnable params, so it's
/// applied parameter-free (`LayerNorm(elementwise_affine: false)`). The reference modulation is
/// SCALE-ONLY (`scale = 1 + adaLN(silu(t))`, output dim = `dim`), with no additive shift.
final class ZImageFinalLayer: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Linear]
    private let eps: Float = 1e-6
    init(dim: Int, outDim: Int) {
        self._linear.wrappedValue = Linear(dim, outDim, bias: true)
        self._adaLNModulation.wrappedValue = [Linear(dim, dim, bias: true)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray, timeEmb: MLXArray) -> MLXArray {
        let scale = expandedDimensions(adaLNModulation[0](silu(timeEmb)), axis: 1)
        // Parameter-free LayerNorm (no affine), then scale-only AdaLN.
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normed = (x - mean) * rsqrt(variance + eps)
        return linear(normed * (1 + scale))
    }
}

/// Final-layer container keyed `2-1` (mirrors `all_x_embedder`): `all_final_layer.2-1.*`.
final class ZImageFinalLayerDict: Module {
    @ModuleInfo(key: "2-1") var layer: ZImageFinalLayer
    init(dim: Int) {
        self._layer.wrappedValue = ZImageFinalLayer(dim: dim, outDim: ZImageConfig.DiT.patchedChannels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, timeEmb: MLXArray) -> MLXArray { layer(x, timeEmb: timeEmb) }
}

/// Adapts a main `ZImageTransformerBlock` to the engine's `StreamableBlock`. The block derives its
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

    @ModuleInfo(key: "layers") var layersList: [ZImageTransformerBlock]
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [ZImageTransformerBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [ZImageTransformerBlock]
    @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "cap_embedder") var capEmbedder: CaptionEmbedder
    @ModuleInfo(key: "all_x_embedder") var allXEmbedder: ZImageXEmbedder
    @ModuleInfo(key: "all_final_layer") var allFinalLayer: ZImageFinalLayerDict
    @ParameterInfo(key: "x_pad_token") var xPadToken: MLXArray
    @ParameterInfo(key: "cap_pad_token") var capPadToken: MLXArray

    // Set during `embed`, used by `unembed` (same generation step).
    private var hp = 0, wp = 0, captionLength = 0
    private var lastTimeEmb: MLXArray?

    public override init() {
        let dim = ZImageConfig.DiT.dim
        let te = TimestepEmbedder(dim: dim)
        self._tEmbedder.wrappedValue = te
        self._capEmbedder.wrappedValue = CaptionEmbedder(inDim: ZImageConfig.DiT.captionDim, dim: dim)
        self._allXEmbedder.wrappedValue = ZImageXEmbedder(dim: dim)
        self._allFinalLayer.wrappedValue = ZImageFinalLayerDict(dim: dim)
        let main = (0..<ZImageConfig.DiT.layers).map { _ in ZImageTransformerBlock(dim: dim, hasAdaLN: true) }
        self._layersList.wrappedValue = main
        self._noiseRefiner.wrappedValue =
            (0..<ZImageConfig.DiT.noiseRefiners).map { _ in ZImageTransformerBlock(dim: dim, hasAdaLN: true) }
        self._contextRefiner.wrappedValue =
            (0..<ZImageConfig.DiT.contextRefiners).map { _ in ZImageTransformerBlock(dim: dim, hasAdaLN: false) }
        self._xPadToken.wrappedValue = zeros([1, dim])
        self._capPadToken.wrappedValue = zeros([1, dim])
        self.blocks = main.enumerated().map { i, b in
            ZImageStreamableBlock(index: i, block: b, timeEmbedder: te, approximateBytes: 120_000_000)
        }
        super.init()
    }

    // image-token count for the current step (set in `embed`, used to slice in `unembed`).
    private var imageTokenCount = 0

    /// Patchify the latent, refine the image and caption streams, and build the single-stream
    /// sequence [image ; caption] for the main layers (reference order: image tokens first).
    public func embed(latent: MLXArray, timestep: MLXArray, conditioning: Conditioning) -> MLXArray {
        let b = latent.dim(0), c = latent.dim(1), h = latent.dim(2), w = latent.dim(3)
        let p = ZImageConfig.DiT.patchSize
        hp = h / p; wp = w / p
        // [B,C,H,W] -> [B, hp*wp, p*p*C]  (channels LAST within each patch token, per reference)
        let patches = latent
            .reshaped([b, c, hp, p, wp, p])
            .transposed(0, 2, 4, 3, 5, 1)        // [b, hp, wp, p1, p2, C]
            .reshaped([b, hp * wp, p * p * c])

        let timeEmb = tEmbedder(timestep)        // [B, dim]
        lastTimeEmb = timeEmb

        var imageTokens = allXEmbedder(patches)              // [B, N, dim]
        for block in noiseRefiner { imageTokens = block(imageTokens, timeEmb: timeEmb) }
        imageTokenCount = imageTokens.dim(1)

        var captionTokens = capEmbedder(conditioning.embeddings)   // [B, L, dim]
        for block in contextRefiner { captionTokens = block(captionTokens, timeEmb: nil) }
        captionLength = captionTokens.dim(1)

        return concatenated([imageTokens, captionTokens], axis: 1) // [B, N+L, dim]
    }

    /// Drop the caption tokens, project the image tokens back to patched latent, and unpatchify.
    public func unembed(_ hidden: MLXArray) -> MLXArray {
        let imageTokens = split(hidden, indices: [imageTokenCount], axis: 1)[0]  // [B, N, dim] (front)
        let timeEmb = lastTimeEmb ?? zeros([imageTokens.dim(0), ZImageConfig.DiT.dim])
        let patches = allFinalLayer(imageTokens, timeEmb: timeEmb)               // [B, N, p*p*C]
        let p = ZImageConfig.DiT.patchSize
        let c = ZImageConfig.DiT.vaeLatentChannels
        let bs = patches.dim(0)
        // [B, N, p*p*C] -> [B, C, H, W]  (inverse of the channel-last patchify)
        return patches
            .reshaped([bs, hp, wp, p, p, c])     // (p1, p2, C) layout
            .transposed(0, 5, 1, 3, 2, 4)        // [b, C, hp, p1, wp, p2]
            .reshaped([bs, c, hp * p, wp * p])
    }
}
