import XCTest
@preconcurrency import MLX
import MLXNN
import DiffusionCore
@testable import ZImageMLX

/// Step 5 of the iPhone block-streaming milestone: prove the architecture's namespace seam (R5) and
/// the resident-submodule loader (`loadShared`) are mechanically correct, WITHOUT a real checkpoint.
///
/// Scope (matching the milestone's R6 caveat): we verify the streaming DATA PATH relative to the
/// resident path — same keys, same tensors — and the per-component routing. Forward numerics / image
/// correctness are explicitly out of scope.
final class ComponentSourceTests: XCTestCase {

    /// `WeightSource` over a `.safetensors`-style dict, COUNTING reads so we can assert which keys
    /// were (and were not) fetched.
    final class CountingDictSource: WeightSource, @unchecked Sendable {
        let arrays: [String: MLXArray]
        let isStreaming: Bool
        let freesOnRelease: Bool
        private(set) var requestedKeys: [String] = []
        init(_ arrays: [String: MLXArray], isStreaming: Bool = true, freesOnRelease: Bool = true) {
            self.arrays = arrays; self.isStreaming = isStreaming; self.freesOnRelease = freesOnRelease
        }
        func tensor(_ key: TensorKey) throws -> MLXArray {
            requestedKeys.append(key.name)
            guard let a = arrays[key.name] else { throw WeightLoadError.missingTensor(key.name) }
            return a
        }
    }

    // MARK: - ZImageComponentSource routing (R5)

    /// The composite routes `"<component>/<rest>"` to the matching sub-source (stripping the prefix),
    /// exposes each sub-source via `subSource`, and surfaces the TRANSFORMER's streaming flags.
    func testComponentSourceRoutesByPrefixAndExposesSubSources() throws {
        let enc = CountingDictSource(["norm.weight": MLXArray([Float(1)])], isStreaming: false, freesOnRelease: false)
        let tx = CountingDictSource(["layers.0.x": MLXArray([Float(2)])], isStreaming: true, freesOnRelease: true)
        let vae = CountingDictSource(["decoder.y": MLXArray([Float(3)])], isStreaming: false, freesOnRelease: false)
        let composite = ZImageComponentSource(
            sources: [.textEncoder: enc, .transformer: tx, .vae: vae])

        // Routed read strips the component prefix and hits the right sub-source.
        XCTAssertEqual(try composite.tensor(TensorKey("text_encoder/norm.weight")).item(Float.self), 1)
        XCTAssertEqual(try composite.tensor(TensorKey("transformer/layers.0.x")).item(Float.self), 2)
        XCTAssertEqual(try composite.tensor(TensorKey("vae/decoder.y")).item(Float.self), 3)
        XCTAssertEqual(enc.requestedKeys, ["norm.weight"], "prefix must be stripped before sub-source")
        XCTAssertEqual(tx.requestedKeys, ["layers.0.x"])

        // subSource hands back the exact instances.
        XCTAssertTrue(composite.subSource(.transformer) as? CountingDictSource === tx)
        XCTAssertTrue(composite.subSource(.vae) as? CountingDictSource === vae)

        // Streaming flags come from the transformer sub-source (the streamed component).
        XCTAssertTrue(composite.isStreaming)
        XCTAssertTrue(composite.freesOnRelease)

        // A key with no recognizable component prefix throws.
        XCTAssertThrowsError(try composite.tensor(TensorKey("no_slash_key")))
        XCTAssertThrowsError(try composite.tensor(TensorKey("unknown/foo")))
    }

    // MARK: - loadShared semantics (skip the streamed prefix; match the resident load)

    /// A small module that mirrors the denoiser shape the architecture loads: a resident quantized
    /// Linear (`proj`) + an array of sub-blocks keyed `layers.N` that the engine streams per step.
    final class MiniBlock: Module {
        @ModuleInfo(key: "lin") var lin: Linear
        init(dim: Int) { self._lin.wrappedValue = Linear(dim, dim, bias: false); super.init() }
    }
    final class MiniDenoiser: Module {
        @ModuleInfo(key: "proj") var proj: Linear      // resident (shared) submodule
        @ModuleInfo(key: "layers") var layers: [MiniBlock]  // streamed per-block
        init(dim: Int, layerCount: Int) {
            self._proj.wrappedValue = Linear(dim, dim, bias: true)
            self._layers.wrappedValue = (0..<layerCount).map { _ in MiniBlock(dim: dim) }
            super.init()
        }
    }

    /// `loadShared(skipPrefixes: ["layers."])` must (a) fill every NON-`layers.` destination param
    /// from the source, matching a whole-tree resident load, and (b) NEVER request a `layers.*` key
    /// (those are owned by per-block streaming). Quantized `proj` gets its `.scales`/`.biases`.
    func testLoadSharedFillsResidentSkipsStreamedPrefix() throws {
        let dim = 64, layerCount = 3
        MLXRandom.seed(11)

        // Build a reference module, randomize + 4-bit quantize, and serialize the whole tree.
        let reference = MiniDenoiser(dim: dim, layerCount: layerCount)
        var randomized: [String: MLXArray] = [:]
        for (path, p) in reference.parameters().flattened() { randomized[path] = MLXRandom.normal(p.shape).asType(p.dtype) }
        reference.update(parameters: ModuleParameters.unflattened(randomized))
        quantize(model: reference, groupSize: 64, bits: 4) { _, layer in layer is Linear }
        MLX.eval(reference)
        var onDisk: [String: MLXArray] = [:]
        for (path, p) in reference.parameters().flattened() { onDisk[path] = p }
        let source = CountingDictSource(onDisk)

        // Load only the resident (non-layers) subtree into a fresh module.
        let target = MiniDenoiser(dim: dim, layerCount: layerCount)
        try ZImageWeights.loadShared(from: source, into: target, skipPrefixes: ["layers."],
                                     groupSize: 64, bits: 4)

        // (a) Every requested key is a non-layers key; no `layers.` key was ever fetched.
        XCTAssertFalse(source.requestedKeys.contains { $0.hasPrefix("layers.") },
                       "loadShared must not read streamed layers.* keys: \(source.requestedKeys.filter { $0.hasPrefix("layers.") })")
        XCTAssertTrue(source.requestedKeys.contains("proj.scales"), "quantized proj must be loaded")

        // (b) The resident `proj` params match the reference exactly (same key-set + values).
        let refParams = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        let dstParams = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        for key in refParams.keys where key.hasPrefix("proj.") {
            guard let d = dstParams[key] else { return XCTFail("target missing resident param \(key)") }
            XCTAssertEqual(d.shape, refParams[key]!.shape, "shape mismatch for \(key)")
            XCTAssertTrue(allClose(d.asType(.float32), refParams[key]!.asType(.float32), atol: 1e-5).item(Bool.self),
                          "value mismatch for resident param \(key)")
        }

        // The streamed `layers.*` subtree must be left ENTIRELY untouched: loadShared skips the
        // prefix, so it neither quantizes those Linears (no `.scales` destination is created) nor
        // fills them. The reference's layers ARE quantized, so the target's layer params still carry
        // the fresh unquantized `lin.weight` and never gained a `lin.scales` — proving the skip.
        XCTAssertTrue(refParams.keys.contains { $0.hasPrefix("layers.") && $0.hasSuffix(".scales") },
                      "sanity: the reference's layers.* should be quantized")
        XCTAssertFalse(dstParams.keys.contains { $0.hasPrefix("layers.") && $0.hasSuffix(".scales") },
                       "loadShared must NOT quantize the skipped layers.* (those stream per-block)")
        XCTAssertTrue(dstParams.keys.contains("layers.0.lin.weight"),
                      "target's skipped layers keep their fresh unquantized weight")
    }

    /// A resident destination param with no matching source tensor must throw (no silent partial load).
    func testLoadSharedThrowsOnMissingResidentTensor() throws {
        let dim = 32
        let target = MiniDenoiser(dim: dim, layerCount: 1)
        let emptySource = CountingDictSource([:])
        XCTAssertThrowsError(try ZImageWeights.loadShared(from: emptySource, into: target, skipPrefixes: ["layers."])) { err in
            guard case WeightLoadError.missingTensor = err else {
                return XCTFail("expected WeightLoadError.missingTensor, got \(err)")
            }
        }
    }
}
