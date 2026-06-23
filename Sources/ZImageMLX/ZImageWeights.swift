@preconcurrency import MLX
import MLXNN
import Foundation

// Loads Z-Image's 4-bit quantized weights (mflux numbered-shard layout) into a Module tree.
//
// On-disk format (per component dir transformer/ text_encoder/ vae/): numbered `*.safetensors`
// shards + `model.safetensors.index.json`. Quantized Linear layers are stored as a `.weight`
// (packed uint32) plus sibling `.scales` and `.biases`; norms/embeddings stay unquantized.
//
// IMPORTANT: the module's parameter paths (`@ModuleInfo(key:)`) must match the safetensors keys
// exactly (e.g. `t_embedder.mlp.0`, `layers.0.attention.to_q`, `all_x_embedder.2-1`). Aligning
// the current module keys to the reference — and adding the noise_refiner/context_refiner blocks —
// is the remaining port work; this loader is the mechanism. Conv weights are PyTorch OIHW and need
// an O,H,W,I transpose for MLX (handled per-key below).
public enum ZImageWeights {

    /// Merge-load every `.safetensors` shard in `dir` into a flat `[key: MLXArray]`.
    public static func tensors(in dir: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default
            .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var merged: [String: MLXArray] = [:]
        for file in files {
            for (key, value) in try loadArrays(url: file) { merged[key] = value }
        }
        return merged
    }

    /// Transpose Conv2d weights from PyTorch OIHW `[out, in, kH, kW]` to MLX OHWI `[out, kH, kW, in]`.
    public static func conv2dWeight(_ w: MLXArray) -> MLXArray {
        w.ndim == 4 ? w.transposed(0, 2, 3, 1) : w
    }

    /// Rename checkpoint keys to the module's parameter paths where they intentionally differ. The
    /// caption embedder is an `nn.Sequential` keyed `cap_embedder.{0,1}` in the checkpoint, but is a
    /// named-key module (`norm`/`proj`) here (a mixed module array can't be updated by MLXNN when
    /// only its non-first element is quantized). Extend this if other containers need bridging.
    public static func canonicalize(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        let remaps = [("cap_embedder.0.", "cap_embedder.norm."), ("cap_embedder.1.", "cap_embedder.proj.")]
        var out = [String: MLXArray](minimumCapacity: tensors.count)
        for (key, value) in tensors {
            var k = key
            for (from, to) in remaps where k.hasPrefix(from) { k = to + k.dropFirst(from.count); break }
            out[k] = value
        }
        return out
    }

    /// Quantize the module's Linear layers that are stored 4-bit (those with a sibling `.scales`),
    /// then load all weights. Conv weights are transposed OIHW→OHWI; tensors with no matching
    /// module parameter (recomputable buffers like `rotary_emb.inv_freq`, converter-only extras)
    /// are dropped so the load is exact.
    public static func load(_ weights: [String: MLXArray], into module: Module,
                            groupSize: Int = 64, bits: Int = 4) {
        let canon = canonicalize(weights)
        // Quantize first so the Quantized{Linear,Embedding} `.scales`/`.biases` params exist as
        // destinations. Both Linear AND Embedding are 4-bit in the checkpoint (embed_tokens too).
        quantize(model: module, groupSize: groupSize, bits: bits) { path, layer in
            (layer is Linear || layer is Embedding) && canon["\(path).scales"] != nil
        }
        // Conv weights in this checkpoint are already MLX-native OHWI — no OIHW->OHWI transpose.
        let valid = Set(module.parameters().flattened().map { $0.0 })
        let filtered = canon.filter { valid.contains($0.key) }
        module.update(parameters: ModuleParameters.unflattened(filtered))
    }
}
