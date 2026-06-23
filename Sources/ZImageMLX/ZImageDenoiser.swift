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

/// Timestep → sinusoidal(256) → MLP → 256-dim embedding (ADALN_EMBED_DIM, the input to every
/// block's AdaLN). Keys: `t_embedder.linear1` (256→1024), `t_embedder.linear2` (1024→256).
final class TimestepEmbedder: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    private let freqDim = ZImageConfig.DiT.adaLNInputDim
    init(dim: Int) {
        let freq = ZImageConfig.DiT.adaLNInputDim
        let hidden = ZImageConfig.DiT.tEmbedderHidden
        self._linear1.wrappedValue = Linear(freq, hidden)
        self._linear2.wrappedValue = Linear(hidden, freq)
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

/// Caption projection: RMSNorm + Linear (2560 → dim). The reference keys these `cap_embedder.{0,1}`
/// (an `nn.Sequential`), but a mixed module ARRAY where only the non-first element is quantized
/// can't be updated by MLXNN (`[.none, .value]` first-element-`.none` is unsupported). So this is a
/// named-key module (`norm`/`proj`) and `ZImageWeights` remaps the checkpoint's `0`/`1` keys on load.
final class CaptionEmbedder: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "proj") var proj: Linear
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
        self._adaLNModulation.wrappedValue = [Linear(ZImageConfig.DiT.adaLNInputDim, dim, bias: true)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray, timeEmb: MLXArray) -> MLXArray {
        let scale = expandedDimensions(adaLNModulation[0](silu(timeEmb.asType(x.dtype))), axis: 1)
        // Parameter-free LayerNorm (no affine) computed in fp32 for stability, then scale-only AdaLN.
        let xf = x.asType(.float32)
        let mean = xf.mean(axis: -1, keepDims: true)
        let variance = xf.variance(axis: -1, keepDims: true)
        let normed = ((xf - mean) * rsqrt(variance + eps)).asType(x.dtype)
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

/// Shared, per-step holder for the UNIFIED-sequence 3D-RoPE tables. `ZImageDenoiser.embed` sets
/// these for the `[image ; caption]` sequence; each main `ZImageStreamableBlock` reads them. (The
/// refiners build their own per-segment tables and pass them directly.)
final class ZImageRopeHolder {
    var cos = MLXArray([Float(0)])
    var sin = MLXArray([Float(0)])
}

/// Adapts a main `ZImageTransformerBlock` to the engine's `StreamableBlock`. The block derives its
/// AdaLN modulation from the timestep via the shared `TimestepEmbedder` and its RoPE tables from the
/// shared `ZImageRopeHolder`. `load`/`release` are no-ops here (weights are resident); the iPhone
/// streaming path will load per-block ranges.
final class ZImageStreamableBlock: StreamableBlock {
    let index: Int
    let approximateBytes: Int64
    private let block: ZImageTransformerBlock
    private let timeEmbedder: TimestepEmbedder
    private let rope: ZImageRopeHolder

    init(index: Int, block: ZImageTransformerBlock, timeEmbedder: TimestepEmbedder,
         rope: ZImageRopeHolder, approximateBytes: Int64) {
        self.index = index
        self.block = block
        self.timeEmbedder = timeEmbedder
        self.rope = rope
        self.approximateBytes = approximateBytes
    }
    func load(from source: WeightSource) throws {}
    func callAsFunction(_ x: MLXArray, conditioning: Conditioning, timestep: MLXArray) -> MLXArray {
        // Z-Image conditions on (1 - sigma): t=0 at the noisy end, t→1 toward clean.
        block(x, timeEmb: timeEmbedder(1.0 - timestep), cos: rope.cos, sin: rope.sin)
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
    // Unified-sequence RoPE tables, shared with the main streamable blocks (set in `embed`).
    private let ropeHolder: ZImageRopeHolder

    public override init() {
        let holder = ZImageRopeHolder()
        self.ropeHolder = holder
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
            ZImageStreamableBlock(index: i, block: b, timeEmbedder: te, rope: holder, approximateBytes: 120_000_000)
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
        let L = conditioning.embeddings.dim(1)

        // 3D-RoPE tables: per-segment (refiners) + unified (main layers). Caption is numbered first
        // on the t-axis; image patches share t=L+1 with (h=row, w=col) in h-outer/w-inner raster.
        let pos = ZImageRoPE.positions(hp: hp, wp: wp, captionLength: L)
        let (imgCos, imgSin) = ZImageRoPE.tables(posT: pos.imgT, posH: pos.imgH, posW: pos.imgW)
        let (capCos, capSin) = ZImageRoPE.tables(posT: pos.capT, posH: pos.capH, posW: pos.capW)
        ropeHolder.cos = concatenated([imgCos, capCos], axis: 0)   // unified [image ; caption]
        ropeHolder.sin = concatenated([imgSin, capSin], axis: 0)

        // [B,C,H,W] -> [B, hp*wp, p*p*C]  (channels LAST within each patch token, per reference)
        let patches = latent
            .reshaped([b, c, hp, p, wp, p])
            .transposed(0, 2, 4, 3, 5, 1)        // [b, hp, wp, p1, p2, C]
            .reshaped([b, hp * wp, p * p * c])

        let timeEmb = tEmbedder(1.0 - timestep)  // condition on (1 - sigma); [B, 256]
        lastTimeEmb = timeEmb

        var imageTokens = allXEmbedder(patches)              // [B, N, dim]
        for block in noiseRefiner { imageTokens = block(imageTokens, timeEmb: timeEmb, cos: imgCos, sin: imgSin) }
        imageTokenCount = imageTokens.dim(1)

        var captionTokens = capEmbedder(conditioning.embeddings)   // [B, L, dim]
        for block in contextRefiner { captionTokens = block(captionTokens, timeEmb: nil, cos: capCos, sin: capSin) }
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
        let velocity = patches
            .reshaped([bs, hp, wp, p, p, c])     // (p1, p2, C) layout
            .transposed(0, 5, 1, 3, 2, 4)        // [b, C, hp, p1, wp, p2]
            .reshaped([bs, c, hp * p, wp * p])
        // Z-Image's raw flow field is negated before the Euler step (`noise_pred = -noise_pred`),
        // so the generic engine update `x + (σ_next − σ)·v` integrates toward the data manifold.
        return -velocity
    }
}
