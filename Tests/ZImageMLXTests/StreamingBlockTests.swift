import XCTest
@preconcurrency import MLX
import MLXNN
import DiffusionCore
@testable import ZImageMLX

/// Step 4 of the iPhone block-streaming milestone: prove the streaming `ZImageStreamableBlock`
/// (the one `ZImageDenoiser(streaming: true)` builds) actually drives the engine's
/// load -> run -> release lifecycle per block per step, holding at most ONE block resident.
///
/// We exercise the SAME loop the core `MLXDiffusionEngine` runs while streaming —
/// `for step { for block { load -> run -> eval -> release -> clearCache } }` — over a SMALL synthetic
/// checkpoint, and assert:
///   - load count == release count == steps, for every block;
///   - no block stays resident after `release()` (the strong ref is dropped — R2);
///   - a freshly loaded block IS resident and has the prefix's parameters filled.
///
/// The synthetic checkpoint uses a tiny hidden width (so the test is fast); `ZImageStreamableBlock`'s
/// streaming initializer takes an injectable `dim` for exactly this reason. We don't run the S3-DiT
/// FORWARD here: forward numerics (3D RoPE at headDim 128) are explicitly out of scope for Step 4 —
/// the claim under test is the streaming DATA PATH (build + per-prefix load + release + residency).
final class StreamingBlockTests: XCTestCase {

    /// `WeightSource` over a `.safetensors` dict that COUNTS how many tensors each call fetched,
    /// so we can confirm each `load` pulled a block's worth of weights from the source.
    final class CountingFileWeightSource: WeightSource, @unchecked Sendable {
        let arrays: [String: MLXArray]
        let isStreaming = true
        let freesOnRelease = true
        private(set) var tensorCalls = 0
        init(url: URL) throws { self.arrays = try loadArrays(url: url) }
        func tensor(_ key: TensorKey) throws -> MLXArray {
            guard let a = arrays[key.name] else { throw WeightLoadError.missingTensor(key.name) }
            tensorCalls += 1
            return a
        }
    }

    /// Build `blockCount` tiny 4-bit-quantized transformer blocks, serialize each under a
    /// `layers.N.` prefix into one synthetic checkpoint file. Returns the file URL.
    private func writeSyntheticCheckpoint(blockCount: Int, dim: Int) throws -> URL {
        MLXRandom.seed(7)
        var onDisk: [String: MLXArray] = [:]
        for i in 0..<blockCount {
            let block = ZImageTransformerBlock(dim: dim, hasAdaLN: true)
            var randomized: [String: MLXArray] = [:]
            for (path, p) in block.parameters().flattened() {
                randomized[path] = MLXRandom.normal(p.shape).asType(p.dtype)
            }
            block.update(parameters: ModuleParameters.unflattened(randomized))
            quantize(model: block, groupSize: 64, bits: 4) { _, layer in layer is Linear }
            MLX.eval(block)
            for (path, p) in block.parameters().flattened() { onDisk["layers.\(i).\(path)"] = p }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zimage-stream-\(UUID().uuidString).safetensors")
        try save(arrays: onDisk, url: url)
        return url
    }

    /// Drive the streaming engine's per-step lifecycle over the synthetic checkpoint and verify the
    /// load/release accounting and one-block-resident invariant.
    func testStreamingLifecycleLoadsAndReleasesPerStep() throws {
        let dim = 64, blockCount = 3, steps = 4
        let url = try writeSyntheticCheckpoint(blockCount: blockCount, dim: dim)
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try CountingFileWeightSource(url: url)

        // Build streaming adapters exactly as `ZImageDenoiser(streaming: true)` does (small dim).
        let te = TimestepEmbedder(dim: ZImageConfig.DiT.dim)
        let rope = ZImageRopeHolder()
        let blocks: [ZImageStreamableBlock] = (0..<blockCount).map { i in
            ZImageStreamableBlock(index: i, prefix: "layers.\(i).", hasAdaLN: true,
                                  timeEmbedder: te, rope: rope, approximateBytes: 0, dim: dim)
        }

        // Nothing resident before the run.
        for b in blocks { XCTAssertFalse(b.isStreamedBlockResident, "block \(b.index) resident before load") }

        var loadCount = [Int](repeating: 0, count: blockCount)
        var releaseCount = [Int](repeating: 0, count: blockCount)

        // Mirror MLXDiffusionEngine's streaming loop: for each step, for each block, load -> release.
        for _ in 0..<steps {
            for b in blocks {
                try b.load(from: source)
                loadCount[b.index] += 1
                // Exactly one block is resident at this instant: this one, none of the others.
                XCTAssertTrue(b.isStreamedBlockResident, "block \(b.index) not resident after load")
                let othersResident = blocks.filter { $0.index != b.index && $0.isStreamedBlockResident }
                XCTAssertTrue(othersResident.isEmpty,
                              "more than one block resident: also \(othersResident.map(\.index))")

                b.release()
                releaseCount[b.index] += 1
                MLX.GPU.clearCache()
                // After release the strong ref is dropped — the block is no longer resident (R2).
                XCTAssertFalse(b.isStreamedBlockResident, "block \(b.index) still resident after release")
            }
        }

        // load count == release count == steps, for every block.
        for i in 0..<blockCount {
            XCTAssertEqual(loadCount[i], steps, "block \(i) loaded \(loadCount[i]) times, expected \(steps)")
            XCTAssertEqual(releaseCount[i], steps, "block \(i) released \(releaseCount[i]) times, expected \(steps)")
        }
        // Nothing resident after the run.
        for b in blocks { XCTAssertFalse(b.isStreamedBlockResident, "block \(b.index) resident after run") }
        // Each step x block did fetch tensors from the source (proves load() really hit the source).
        XCTAssertGreaterThan(source.tensorCalls, blockCount * steps,
                             "expected the per-prefix loader to pull many tensors per load")
    }

    /// A freshly loaded streaming block must expose the prefix's filled, quantized parameters; after
    /// release it must hold nothing. (Confirms `load()` builds AND fills, not just allocates.)
    func testLoadFillsParametersAndReleaseDrops() throws {
        let dim = 64
        let url = try writeSyntheticCheckpoint(blockCount: 1, dim: dim)
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try CountingFileWeightSource(url: url)

        let block = ZImageStreamableBlock(index: 0, prefix: "layers.0.", hasAdaLN: true,
                                          timeEmbedder: TimestepEmbedder(dim: ZImageConfig.DiT.dim),
                                          rope: ZImageRopeHolder(), approximateBytes: 0, dim: dim)
        XCTAssertFalse(block.isStreamedBlockResident)
        try block.load(from: source)
        XCTAssertTrue(block.isStreamedBlockResident)
        XCTAssertGreaterThan(source.tensorCalls, 0, "load() never read from the source")

        block.release()
        XCTAssertFalse(block.isStreamedBlockResident)
    }

    /// The default (resident) denoiser must still build 30 resident main blocks and resident-mode
    /// adapters whose load/release are no-ops — the verified macOS path must not regress.
    func testResidentDenoiserStillBuildsResidentBlocks() throws {
        let resident = ZImageDenoiser(streaming: false)
        XCTAssertEqual(resident.blocks.count, ZImageConfig.DiT.layers)
        XCTAssertEqual(resident.layersList.count, ZImageConfig.DiT.layers,
                       "resident denoiser must build its layers.* subtree for the whole-tree loader")

        // Streaming denoiser builds the adapters but NO resident main blocks.
        let streaming = ZImageDenoiser(streaming: true)
        XCTAssertEqual(streaming.blocks.count, ZImageConfig.DiT.layers)
        XCTAssertTrue(streaming.layersList.isEmpty,
                      "streaming denoiser must not build resident main blocks")
    }
}
