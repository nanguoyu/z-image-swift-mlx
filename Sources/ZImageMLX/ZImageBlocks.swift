@preconcurrency import MLX
import MLXNN
import MLXFast
import Foundation

// S3-DiT building blocks, grounded in the Z-Image reference (see IMPLEMENTATION.md).
//
// NOTE: layer shapes and weight keys (`@ModuleInfo(key:)`) match the reference state_dict so
// weights load directly, and these compile against MLXNN. The FORWARD numerics — especially the
// 3D-axes RoPE (currently a 1D placeholder) and the AdaLN application — still require parity
// validation against the Python reference on a GPU. Nothing here has been run.

/// SwiGLU feed-forward: `w2(silu(w1 x) * w3 x)`, no bias.
final class ZImageFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int = ZImageConfig.DiT.dim, hidden: Int = ZImageConfig.DiT.ffnHidden) {
        self._w1.wrappedValue = Linear(dim, hidden, bias: false)
        self._w2.wrappedValue = Linear(hidden, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// Grouped-query attention with per-head QK RMSNorm and RoPE.
final class ZImageAttention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear   // reference key `to_out.0` → remap on load
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    let rope: RoPE

    init(dim: Int = ZImageConfig.DiT.dim,
         heads: Int = ZImageConfig.DiT.heads,
         headDim: Int = ZImageConfig.DiT.headDim) {
        self.heads = heads
        self.headDim = headDim
        self._toQ.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toK.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toV.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toOut.wrappedValue = Linear(heads * headDim, dim, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: ZImageConfig.DiT.rmsEps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: ZImageConfig.DiT.rmsEps)
        // Placeholder 1D RoPE; the reference uses 3D-axes RoPE (see IMPLEMENTATION.md).
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ZImageConfig.DiT.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        // [B, N, H, Dh]
        var q = toQ(x).reshaped([b, n, heads, headDim])
        var k = toK(x).reshaped([b, n, heads, headDim])
        let v = toV(x).reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
        // Per-head QK RMSNorm (normalizes the last axis = headDim).
        q = normQ(q).transposed(0, 2, 1, 3)
        k = normK(k).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        let scale = 1.0 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                    scale: scale, mask: nil)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim])
        return toOut(merged)
    }
}

/// One S3-DiT block: AdaLN-modulated attention + SwiGLU FFN (Lumina-NextDiT style).
final class ZImageTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: ZImageAttention
    @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear

    private let dim: Int

    init(dim: Int = ZImageConfig.DiT.dim) {
        self.dim = dim
        self._attention.wrappedValue = ZImageAttention(dim: dim)
        self._feedForward.wrappedValue = ZImageFeedForward(dim: dim)
        self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._adaLNModulation.wrappedValue = Linear(dim, 4 * dim, bias: true)
        super.init()
    }

    /// `timeEmb` is the timestep embedding `[B, dim]`. AdaLN produces (scale, gate) for the
    /// attention and FFN sub-layers. NOTE: exact modulation formula needs reference validation.
    func callAsFunction(_ x: MLXArray, timeEmb: MLXArray) -> MLXArray {
        let mod = adaLNModulation(silu(timeEmb))            // [B, 4*dim]
        let parts = split(mod, parts: 4, axis: -1)          // 4 × [B, dim]
        let scaleAttn = expandedDimensions(parts[0], axis: 1)
        let gateAttn = expandedDimensions(parts[1], axis: 1)
        let scaleFFN = expandedDimensions(parts[2], axis: 1)
        let gateFFN = expandedDimensions(parts[3], axis: 1)

        var h = x + gateAttn * attentionNorm2(attention(attentionNorm1(x) * (1 + scaleAttn)))
        h = h + gateFFN * ffnNorm2(feedForward(ffnNorm1(h) * (1 + scaleFFN)))
        return h
    }
}
