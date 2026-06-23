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

    /// Quantize the module's Linear layers that are stored 4-bit (those with a sibling `.scales`),
    /// then load all weights. Keys in `weights` must match the module's parameter paths.
    public static func load(_ weights: [String: MLXArray], into module: Module,
                            groupSize: Int = 64, bits: Int = 4) {
        quantize(model: module, groupSize: groupSize, bits: bits) { path, layer in
            layer is Linear && weights["\(path).scales"] != nil
        }
        module.update(parameters: ModuleParameters.unflattened(weights))
    }
}
