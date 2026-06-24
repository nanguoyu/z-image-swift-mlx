@preconcurrency import MLX
import Foundation
import DiffusionCore

/// Resolves the Z-Image 3-namespace collision (R5) at the `WeightSource` seam.
///
/// `MLXDiffusionEngine` hands the architecture exactly ONE flat `WeightSource`, but Z-Image is three
/// separate component trees — `text_encoder/`, `transformer/`, `vae/` — whose key spaces collide on
/// generic names (`norm.weight`, `0.scales`, …). A single flat source can't represent them without a
/// global key remap; instead this is a COMPOSITE source that owns one sub-`WeightSource` per
/// component folder. `ZImageArchitecture` downcasts the engine's `source` back to this type and pulls
/// the per-component sub-source it needs for each phase (encode → transformer → vae), so each
/// component is opened as its own source and the namespaces never collide.
///
/// `tensor(_:)` routes an explicit `"<component>/"` key prefix to that component, and serves any BARE
/// key from the transformer sub-source — because the generic engine streams the denoiser blocks by
/// calling `source.tensor("layers.0.…")` on this composite directly. `encode`/`decode` use
/// `subSource(_:)` instead so they get a clean per-component source downstream.
///
/// `isStreaming` / `freesOnRelease` are taken from the TRANSFORMER sub-source — that is the one the
/// engine streams block-by-block, and the one whose `freesOnRelease` gates the streaming residency
/// plan. (The text encoder and VAE are loaded resident in a single pass, so their flags don't drive
/// the streaming decision.)
public final class ZImageComponentSource: WeightSource, @unchecked Sendable {

    /// The three Z-Image component trees, each backed by its own `WeightSource`.
    public enum Component: String, CaseIterable, Sendable {
        case textEncoder = "text_encoder"
        case transformer = "transformer"
        case vae = "vae"
    }

    public enum SourceError: Error, CustomStringConvertible {
        case unknownComponent(String)
        case missingComponentFolder(String)
        case noSafetensors(String)
        public var description: String {
            switch self {
            case .unknownComponent(let k):
                return "ZImageComponentSource: key '\(k)' has no '<component>/' prefix (expected one of \(Component.allCases.map(\.rawValue)))"
            case .missingComponentFolder(let c):
                return "ZImageComponentSource: component folder '\(c)' does not exist in the model directory"
            case .noSafetensors(let c):
                return "ZImageComponentSource: no .safetensors files in component folder '\(c)'"
            }
        }
    }

    private let lock = NSLock()
    private var sources: [Component: any WeightSource]
    /// The tokenizer folder, passed through so the architecture can resolve the Qwen3 tokenizer
    /// without a hub round-trip. `nil` if the model directory has no `tokenizer/` folder.
    public let tokenizerDirectory: URL?

    public var isStreaming: Bool { lock.withLock { sources[.transformer]?.isStreaming ?? false } }
    public var freesOnRelease: Bool { lock.withLock { sources[.transformer]?.freesOnRelease ?? false } }

    /// Build from explicit per-component sub-sources (used by tests and advanced callers).
    public init(sources: [Component: any WeightSource], tokenizerDirectory: URL? = nil) {
        self.sources = sources
        self.tokenizerDirectory = tokenizerDirectory
    }

    /// The sub-source for one component, or `nil` if this composite wasn't given that component.
    public func subSource(_ component: Component) -> (any WeightSource)? { lock.withLock { sources[component] } }

    /// Drop a component's sub-source after its phase is done so its eagerly-held weights free. The
    /// text encoder's `SafetensorsWeightSource` otherwise holds ~2 GB of arrays for the whole run
    /// (the composite is retained by the engine through decode); releasing it after `encode` reclaims
    /// that memory before the transformer streams.
    public func releaseComponent(_ component: Component) {
        lock.withLock { _ = sources.removeValue(forKey: component) }
        MLX.GPU.clearCache()
    }

    /// Resolve a tensor by key. An explicit `"<component>/"` prefix routes to that component. A BARE
    /// key (no `"<component>/"` prefix) is served from the TRANSFORMER sub-source: the generic
    /// `MLXDiffusionEngine` streams the denoiser blocks by calling `source.tensor("layers.0.…")` on
    /// this composite directly (it has no notion of the component split), and the transformer is the
    /// streamed component. `encode`/`decode` resolve their component via `subSource(_:)` and never
    /// reach this method, so bare keys here are always transformer keys — no collision.
    public func tensor(_ key: TensorKey) throws -> MLXArray {
        if let slash = key.name.firstIndex(of: "/"),
           let component = Component(rawValue: String(key.name[..<slash])),
           let sub = lock.withLock({ sources[component] }) {
            let rest = String(key.name[key.name.index(after: slash)...])
            return try sub.tensor(TensorKey(rest))
        }
        guard let transformer = lock.withLock({ sources[.transformer] }) else {
            throw SourceError.unknownComponent(key.name)
        }
        return try transformer.tensor(key)
    }
}

public extension ZImageComponentSource {

    /// Open a Z-Image model directory (`text_encoder/`, `transformer/`, `vae/`, `tokenizer/`) as a
    /// composite source. `streaming == true` opens each component with `RangedFileWeightSource`
    /// (`pread` on demand, `freesOnRelease == true`) so the engine's block-streaming path actually
    /// reclaims memory; `false` opens `SafetensorsWeightSource` (mmap, resident).
    ///
    /// This is the iPhone partial-load entry point: the app builds one of these from the downloaded
    /// model folder and hands it to `MLXDiffusionEngine.load(...)`.
    static func open(modelDirectory: URL, streaming: Bool) throws -> ZImageComponentSource {
        var sources: [Component: any WeightSource] = [:]
        for component in Component.allCases {
            let dir = modelDirectory.appendingPathComponent(component.rawValue)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
                throw SourceError.missingComponentFolder(component.rawValue)
            }
            let files = try FileManager.default
                .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard !files.isEmpty else { throw SourceError.noSafetensors(component.rawValue) }
            // The transformer is the only component the engine streams; open it streaming when asked.
            // Text encoder + VAE are loaded resident in one pass regardless, so keep them mmap-backed.
            let streamThis = streaming && component == .transformer
            let sub: any WeightSource = streamThis
                ? try RangedFileWeightSource(files: files, isStreaming: true)
                : try SafetensorsWeightSource(files: files, isStreaming: false)
            sources[component] = sub
        }
        let tok = modelDirectory.appendingPathComponent("tokenizer")
        var tokIsDir: ObjCBool = false
        let tokExists = FileManager.default.fileExists(atPath: tok.path, isDirectory: &tokIsDir) && tokIsDir.boolValue
        return ZImageComponentSource(sources: sources, tokenizerDirectory: tokExists ? tok : nil)
    }
}
