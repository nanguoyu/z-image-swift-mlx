@preconcurrency import MLX
import MLXNN
import DiffusionCore
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

    /// Streaming counterpart of `load`, scoped to ONE key prefix (e.g. `"layers.0."`). Loads only
    /// the tensors under `prefix` from any `WeightSource` (mmap, ranged-pread, hybrid) into a single
    /// block `module`, reusing the SAME quantize→canonicalize→filter→update pipeline as the whole-tree
    /// loader — just keyed relative to the block instead of the full denoiser tree.
    ///
    /// The whole-tree loader can quantize-by-probing a flat merged dict; here there is no merged dict,
    /// so we discover the per-block layout by *probing the source*: a Linear/Embedding is 4-bit iff
    /// its sibling `<path>.scales` tensor exists under `prefix`. We quantize those destinations FIRST
    /// (so `.scales`/`.biases` params exist — R4), then pull every destination parameter by exact
    /// `TensorKey` (`source.tensor(prefix + relPath)`), never a whole-shard merge-load. Tensors with no
    /// matching destination param are simply never requested, so the load stays exact.
    ///
    /// Probe failures (a tensor the source doesn't have) are treated as "absent" so we can cheaply ask
    /// "is `<path>.scales` present?"; a *destination* parameter that fails to fetch throws.
    public static func loadBlock(prefix: String, from source: WeightSource, into module: Module,
                                 groupSize: Int = 64, bits: Int = 4) throws {
        // Returns the tensor for `prefix+relKey` if the source has it, else nil (probe).
        func probe(_ relKey: String) -> MLXArray? {
            try? source.tensor(TensorKey(prefix + relKey))
        }
        // Quantize the destinations that are 4-bit on disk (sibling `.scales` present under prefix),
        // mirroring the whole-tree loader's `canon["\(path).scales"] != nil` predicate.
        quantize(model: module, groupSize: groupSize, bits: bits) { path, layer in
            (layer is Linear || layer is Embedding) && probe("\(path).scales") != nil
        }
        // Now enumerate the (post-quantize) destination parameter paths and fetch each by exact key.
        var collected: [String: MLXArray] = [:]
        for (relPath, _) in module.parameters().flattened() {
            guard let t = try? source.tensor(TensorKey(prefix + relPath)) else {
                // A destination param with no source tensor is a layout mismatch — surface it.
                throw WeightLoadError.missingTensor(prefix + relPath)
            }
            collected[relPath] = t
        }
        // `canonicalize` is a no-op for the block-local keys here (its only remap targets
        // `cap_embedder.*`, which lives outside a transformer block), but apply it for symmetry so the
        // two loaders share one pipeline.
        let canon = canonicalize(collected)
        let valid = Set(module.parameters().flattened().map { $0.0 })
        let filtered = canon.filter { valid.contains($0.key) }
        module.update(parameters: ModuleParameters.unflattened(filtered))
    }

    /// Load every destination parameter of `module` from `source` by exact `TensorKey`, EXCEPT those
    /// whose path begins with any of `skipPrefixes`. This is the per-tensor (source-agnostic) twin of
    /// the whole-tree `load`, used to fill the streaming denoiser's RESIDENT submodules (embedders,
    /// refiners, pad tokens, final layer) from the transformer component while leaving the 30 main
    /// `layers.*` blocks to stream per step.
    ///
    /// Quantization is discovered by probing (sibling `<path>.scales` present), exactly like
    /// `loadBlock`, so quantized Linears/Embeddings get their `.scales`/`.biases` destinations before
    /// `update` (R4). The checkpoint's `cap_embedder.{0,1}` → `cap_embedder.{norm,proj}` remap is
    /// applied to BOTH the probe and the fetch, so the renamed module params resolve to the on-disk
    /// keys. A destination param with no matching source tensor throws (no silent partial load).
    public static func loadShared(from source: WeightSource, into module: Module,
                                  skipPrefixes: [String], groupSize: Int = 64, bits: Int = 4) throws {
        // Reverse of `canonicalize`: map a module parameter path back to its on-disk checkpoint key
        // so we can fetch/probe it from the source (which is keyed by the original checkpoint names).
        func diskKey(_ modulePath: String) -> String {
            let remaps = [("cap_embedder.norm.", "cap_embedder.0."), ("cap_embedder.proj.", "cap_embedder.1.")]
            for (from, to) in remaps where modulePath.hasPrefix(from) { return to + modulePath.dropFirst(from.count) }
            return modulePath
        }
        func isSkipped(_ path: String) -> Bool { skipPrefixes.contains { path.hasPrefix($0) } }

        // Quantize the resident-submodule Linears/Embeddings that are 4-bit on disk (probe sibling
        // `.scales` via the disk key), skipping anything under a streamed prefix.
        quantize(model: module, groupSize: groupSize, bits: bits) { path, layer in
            guard (layer is Linear || layer is Embedding), !isSkipped(path) else { return false }
            return (try? source.tensor(TensorKey(diskKey("\(path).scales")))) != nil
        }
        // Fetch every non-skipped destination parameter by exact key.
        var collected: [String: MLXArray] = [:]
        for (path, _) in module.parameters().flattened() {
            if isSkipped(path) { continue }
            guard let t = try? source.tensor(TensorKey(diskKey(path))) else {
                throw WeightLoadError.missingTensor(diskKey(path))
            }
            collected[path] = t
        }
        module.update(parameters: ModuleParameters.unflattened(collected))
    }
}

/// Errors raised by the per-block streaming loader.
public enum WeightLoadError: Error, CustomStringConvertible {
    /// A destination module parameter had no corresponding tensor in the `WeightSource`.
    case missingTensor(String)
    public var description: String {
        switch self {
        case .missingTensor(let key): return "WeightSource is missing tensor for key '\(key)'"
        }
    }
}
