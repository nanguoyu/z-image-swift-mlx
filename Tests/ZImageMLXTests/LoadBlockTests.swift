import XCTest
@preconcurrency import MLX
import MLXNN
import DiffusionCore
@testable import ZImageMLX

/// Step 3 of the iPhone block-streaming milestone: prove the per-prefix single-block loader
/// (`ZImageWeights.loadBlock`) reconstructs a transformer block's parameters identically to the
/// resident whole-tree loader (`ZImageWeights.load`) for the same key prefix, and that it pulls
/// every tensor it needs by exact `TensorKey` from a generic `WeightSource`.
///
/// We don't depend on core's `RangedFileWeightSource` here (it isn't on the resolved core branch
/// yet); a tiny in-test `WeightSource` backed by a `.safetensors` file exercises exactly the same
/// `source.tensor(_:)` contract `loadBlock` relies on, so the mechanical claims hold for any source.
final class LoadBlockTests: XCTestCase {

    /// In-test `WeightSource`: materializes tensors by name from a single `.safetensors` file via
    /// MLX's own `loadArrays`. Stands in for `RangedFileWeightSource` for the generic-protocol path.
    struct FileWeightSource: WeightSource {
        let arrays: [String: MLXArray]
        let isStreaming = true
        let freesOnRelease = true
        init(url: URL) throws { self.arrays = try loadArrays(url: url) }
        func tensor(_ key: TensorKey) throws -> MLXArray {
            guard let a = arrays[key.name] else {
                throw WeightLoadError.missingTensor(key.name)
            }
            return a
        }
    }

    /// Build a 4-bit-quantized "checkpoint" for ONE main block, written to disk under a `layers.0.`
    /// prefix, then compare: (resident) whole-tree `load` of the prefix-stripped dict vs. (streaming)
    /// `loadBlock` straight from a `WeightSource`. Every parameter must match (allClose).
    func testLoadBlockMatchesResidentLoadForPrefix() throws {
        let prefix = "layers.0."

        // 1. A reference block with deterministic random fp32 params, quantized 4-bit exactly the way
        //    the on-disk checkpoint stores main blocks (Linear layers -> .weight/.scales/.biases).
        MLXRandom.seed(42)
        let reference = ZImageTransformerBlock(hasAdaLN: true)
        // Randomize the params first (so quantization has nontrivial values to round-trip).
        var randomized: [String: MLXArray] = [:]
        for (path, p) in reference.parameters().flattened() {
            randomized[path] = MLXRandom.normal(p.shape).asType(p.dtype)
        }
        reference.update(parameters: ModuleParameters.unflattened(randomized))
        quantize(model: reference, groupSize: 64, bits: 4) { _, layer in layer is Linear }
        MLX.eval(reference)

        // 2. Serialize the quantized reference's full parameter set under the `layers.0.` prefix.
        var onDisk: [String: MLXArray] = [:]
        for (path, p) in reference.parameters().flattened() { onDisk[prefix + path] = p }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zimage-block-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        // 3. RESIDENT path: strip the prefix into a flat dict and run the whole-tree loader.
        let residentArrays = try loadArrays(url: url)
        var stripped: [String: MLXArray] = [:]
        for (k, v) in residentArrays where k.hasPrefix(prefix) {
            stripped[String(k.dropFirst(prefix.count))] = v
        }
        let residentBlock = ZImageTransformerBlock(hasAdaLN: true)
        try ZImageWeights.load(stripped, into: residentBlock, groupSize: 64, bits: 4)
        MLX.eval(residentBlock)

        // 4. STREAMING path: per-prefix loader straight from a WeightSource.
        let source = try FileWeightSource(url: url)
        let streamingBlock = ZImageTransformerBlock(hasAdaLN: true)
        try ZImageWeights.loadBlock(prefix: prefix, from: source, into: streamingBlock,
                                    groupSize: 64, bits: 4)
        MLX.eval(streamingBlock)

        // 5. Same parameter set, same values.
        let residentParams = Dictionary(uniqueKeysWithValues: residentBlock.parameters().flattened())
        let streamingParams = Dictionary(uniqueKeysWithValues: streamingBlock.parameters().flattened())
        XCTAssertEqual(Set(streamingParams.keys), Set(residentParams.keys),
                       "streaming and resident loads must produce the same parameter paths")
        XCTAssertFalse(streamingParams.isEmpty, "loader produced no parameters")

        for (path, residentVal) in residentParams {
            let streamingVal = try XCTUnwrap(streamingParams[path], "missing \(path) in streaming load")
            XCTAssertEqual(streamingVal.shape, residentVal.shape, "shape mismatch at \(path)")
            XCTAssertEqual(streamingVal.dtype, residentVal.dtype, "dtype mismatch at \(path)")
            let close = allClose(streamingVal.asType(.float32), residentVal.asType(.float32),
                                 atol: 1e-5).item(Bool.self)
            XCTAssertTrue(close, "value mismatch at \(path)")
        }

        // 6. The streaming load must have quantized the Linear destinations (so .scales/.biases exist).
        XCTAssertTrue(streamingParams.keys.contains("attention.to_q.scales"),
                      "quantized Linear destination should expose .scales")
        XCTAssertTrue(streamingParams.keys.contains("attention.to_q.biases"),
                      "quantized Linear destination should expose .biases")
    }

    /// A destination parameter with no matching source tensor must throw, not silently load partial.
    func testLoadBlockThrowsOnMissingTensor() throws {
        let source = FakeEmptySource()
        let block = ZImageTransformerBlock(hasAdaLN: true)
        XCTAssertThrowsError(try ZImageWeights.loadBlock(prefix: "layers.0.", from: source, into: block))
    }

    /// The whole-tree `load` must also assert coverage: a checkpoint missing ONE model parameter
    /// would leave that param at random init (silent garbage). Build a complete quantized checkpoint
    /// for one block, drop a single key, and assert `load` throws `uncoveredParameters` naming it
    /// — proving the guard fires (and that a COMPLETE checkpoint still loads without throwing).
    func testWholeTreeLoadThrowsOnUncoveredParameter() throws {
        // 1. A reference block, quantized exactly like the on-disk checkpoint (Linear -> 4-bit).
        MLXRandom.seed(7)
        let reference = ZImageTransformerBlock(hasAdaLN: true)
        var randomized: [String: MLXArray] = [:]
        for (path, p) in reference.parameters().flattened() {
            randomized[path] = MLXRandom.normal(p.shape).asType(p.dtype)
        }
        reference.update(parameters: ModuleParameters.unflattened(randomized))
        quantize(model: reference, groupSize: 64, bits: 4) { _, layer in layer is Linear }

        // 2. Flatten the quantized reference's full (post-quantize) parameter set — this is the
        //    complete checkpoint for the block.
        var complete: [String: MLXArray] = [:]
        for (path, p) in reference.parameters().flattened() { complete[path] = p }

        // 3a. The COMPLETE checkpoint loads cleanly (no false positive on a legitimate load).
        let okBlock = ZImageTransformerBlock(hasAdaLN: true)
        XCTAssertNoThrow(try ZImageWeights.load(complete, into: okBlock, groupSize: 64, bits: 4))

        // 3b. Drop ONE key → load must throw, naming the uncovered parameter.
        let droppedKey = "attention.to_q.weight"
        XCTAssertNotNil(complete[droppedKey], "test precondition: \(droppedKey) should exist")
        var missingOne = complete
        missingOne.removeValue(forKey: droppedKey)

        let brokenBlock = ZImageTransformerBlock(hasAdaLN: true)
        XCTAssertThrowsError(
            try ZImageWeights.load(missingOne, into: brokenBlock, groupSize: 64, bits: 4)
        ) { error in
            guard case let WeightLoadError.uncoveredParameters(keys) = error else {
                return XCTFail("expected .uncoveredParameters, got \(error)")
            }
            XCTAssertTrue(keys.contains(droppedKey),
                          "uncovered list should name the dropped key \(droppedKey); got \(keys)")
        }
    }

    /// `WeightSource` that has no tensors at all — every `tensor(_:)` throws.
    struct FakeEmptySource: WeightSource {
        let isStreaming = true
        let freesOnRelease = true
        func tensor(_ key: TensorKey) throws -> MLXArray {
            throw WeightLoadError.missingTensor(key.name)
        }
    }
}
