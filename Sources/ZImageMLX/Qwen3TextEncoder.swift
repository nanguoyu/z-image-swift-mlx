@preconcurrency import MLX
import MLXNN
import MLXFast
import Foundation

// Qwen3-4B text encoder, reimplemented in MLX. swift-transformers is CoreML-only and can't
// expose hidden states, so the encoder is native MLX; swift-transformers is used only for the
// tokenizer + chat template. Z-Image uses the SECOND-TO-LAST hidden state (dim 2560) as caption
// conditioning. Standard Qwen3 decoder: GQA with per-head q/k RMSNorm, RoPE, SwiGLU.
//
// NOTE: structure + weight keys match the reference; forward numerics need GPU parity
// validation. 4-bit weights are loaded by quantizing the Linear layers at load time.

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, intermediate: Int) {
        self._gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._up.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._down.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class Qwen3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPE

    init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, ropeTheta: Float, eps: Float) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        self._qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        // Per-head q/k RMSNorm (Qwen3), then transpose to [B, H, N, Dh].
        var q = qNorm(qProj(x).reshaped([b, n, heads, headDim])).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped([b, n, kvHeads, headDim])).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([b, n, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        // GQA: scaledDotProductAttention broadcasts the kv heads when heads % kvHeads == 0.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim]))
    }
}

final class Qwen3Layer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    override init() {
        let c = ZImageConfig.TextEncoder.self
        self._selfAttn.wrappedValue = Qwen3Attention(
            hidden: c.hidden, heads: c.heads, kvHeads: c.kvHeads, headDim: c.headDim,
            ropeTheta: c.ropeTheta, eps: c.rmsEps)
        self._mlp.wrappedValue = Qwen3MLP(hidden: c.hidden, intermediate: c.ffnHidden)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.rmsEps)
        self._postNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.rmsEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + selfAttn(inputNorm(x), mask: mask)
        h = h + mlp(postNorm(h))
        return h
    }
}

/// Z-Image's caption encoder. Runs the prompt through Qwen3-4B and returns the second-to-last
/// hidden state `[B, N, 2560]`. The converted checkpoint strips the HF `model.` prefix, so the
/// embeddings/layers/norm live at the top level here.
public final class Qwen3TextEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Qwen3Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public override init() {
        let c = ZImageConfig.TextEncoder.self
        self._embedTokens.wrappedValue = Embedding(embeddingCount: 151_936, dimensions: c.hidden)
        self.layers = (0..<c.layers).map { _ in Qwen3Layer() }
        self._norm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.rmsEps)
        super.init()
    }

    /// `tokens` is `[B, N]` Int32 token ids. Returns the layer[-2] hidden state.
    public func hiddenStates(_ tokens: MLXArray) -> MLXArray {
        var h = embedTokens(tokens)
        let mask = MultiHeadAttention.createAdditiveCausalMask(tokens.dim(1)).asType(h.dtype)
        let target = layers.count - 2   // output of the second-to-last layer == hidden[-2]
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask)
            if i == target { return h }
        }
        return h
    }
}
