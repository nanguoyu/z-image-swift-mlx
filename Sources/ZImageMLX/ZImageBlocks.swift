@preconcurrency import MLX
import MLXNN
import MLXFast
import Foundation

// S3-DiT building blocks, grounded in the Z-Image reference (see IMPLEMENTATION.md). Layer
// shapes and weight keys (`@ModuleInfo(key:)`) match the reference state_dict so weights load
// directly. FORWARD numerics — the 3D-axes RoPE (1D placeholder), AdaLN application, and the
// single-stream refiner flow — still require GPU parity validation.

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
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

/// Attention with per-head QK RMSNorm and RoPE. `to_out` is a 1-element list so its key is
/// `to_out.0` (matching the reference `nn.Sequential` indexing).
final class ZImageAttention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(dim: Int = ZImageConfig.DiT.dim,
         heads: Int = ZImageConfig.DiT.heads,
         headDim: Int = ZImageConfig.DiT.headDim) {
        self.heads = heads; self.headDim = headDim
        self._toQ.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toK.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toV.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toOut.wrappedValue = [Linear(heads * headDim, dim, bias: false)]
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: ZImageConfig.DiT.rmsEps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: ZImageConfig.DiT.rmsEps)
        super.init()
    }

    /// `cos`/`sin` are the `[N, 64]` 3D-RoPE tables for the tokens being processed (per-segment in
    /// the refiners, unified in the main layers). Applied to q/k after QK-RMSNorm; v is untouched.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        var q = normQ(toQ(x).reshaped([b, n, heads, headDim])).transposed(0, 2, 1, 3)
        var k = normK(toK(x).reshaped([b, n, heads, headDim])).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
        q = ZImageRoPE.apply(q, cos: cos, sin: sin)
        k = ZImageRoPE.apply(k, cos: cos, sin: sin)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: nil)
        return toOut[0](out.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim]))
    }
}

/// One S3-DiT block. `layers` and `noise_refiner` blocks carry AdaLN modulation (driven by the
/// timestep embedding); `context_refiner` blocks do not (passed `hasAdaLN: false`), so they emit
/// no `adaLN_modulation` key.
final class ZImageTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: ZImageAttention
    @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Linear]

    init(dim: Int = ZImageConfig.DiT.dim, hasAdaLN: Bool = true) {
        self._attention.wrappedValue = ZImageAttention(dim: dim)
        self._feedForward.wrappedValue = ZImageFeedForward(dim: dim)
        self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: ZImageConfig.DiT.rmsEps)
        self._adaLNModulation.wrappedValue =
            hasAdaLN ? [Linear(ZImageConfig.DiT.adaLNInputDim, 4 * dim, bias: true)] : []
        super.init()
    }

    /// `timeEmb` is the timestep embedding `[B, dim]`; ignored by context_refiner blocks. `cos`/`sin`
    /// are the 3D-RoPE tables for the tokens being processed. NOTE: exact AdaLN numerics need parity.
    func callAsFunction(_ x: MLXArray, timeEmb: MLXArray?, cos: MLXArray, sin: MLXArray) -> MLXArray {
        if let ada = adaLNModulation.first, let t = timeEmb {
            let parts = split(ada(silu(t)), parts: 4, axis: -1)
            let scaleAttn = expandedDimensions(parts[0], axis: 1)
            let gateAttn = expandedDimensions(parts[1], axis: 1)
            let scaleFFN = expandedDimensions(parts[2], axis: 1)
            let gateFFN = expandedDimensions(parts[3], axis: 1)
            var h = x + gateAttn * attentionNorm2(attention(attentionNorm1(x) * (1 + scaleAttn), cos: cos, sin: sin))
            h = h + gateFFN * ffnNorm2(feedForward(ffnNorm1(h) * (1 + scaleFFN)))
            return h
        } else {
            var h = x + attentionNorm2(attention(attentionNorm1(x), cos: cos, sin: sin))
            h = h + ffnNorm2(feedForward(ffnNorm1(h)))
            return h
        }
    }
}
