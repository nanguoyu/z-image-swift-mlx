import Foundation
import Hub

/// A model in the built-in catalog (Hugging Face repo + display metadata).
public struct ZImageCatalogModel: Identifiable, Sendable, Hashable {
    public let id: String          // Hugging Face repo id
    public let name: String
    public let detail: String
    public let approximateBytes: Int64
    public init(id: String, name: String, detail: String, approximateBytes: Int64) {
        self.id = id; self.name = name; self.detail = detail; self.approximateBytes = approximateBytes
    }
}

/// The (currently single-entry) Z-Image catalog. Open weights, downloaded in-app.
public enum ZImageCatalog {
    public static let turboQ4 = ZImageCatalogModel(
        id: "deepsweet/Z-Image-Turbo-6B-MLX-Q4",
        name: "Z-Image Turbo (6B)",
        detail: "4-bit · 8-step · S3-DiT + Qwen3-4B",
        approximateBytes: 5_900_000_000)
    public static let all = [turboQ4]
}

/// Downloads model snapshots from Hugging Face into a chosen base directory (via swift-transformers
/// `HubApi`). Snapshots are idempotent — an already-downloaded model verifies quickly and is reused.
public struct ModelDownloader: Sendable {
    public let downloadBase: URL
    public init(downloadBase: URL) { self.downloadBase = downloadBase }

    /// Where `repoId` materializes (matches `HubApi`'s `downloadBase/models/{repoId}`).
    public func localURL(repoId: String) -> URL {
        downloadBase.appending(component: "models").appending(component: repoId)
    }

    /// "Already fully downloaded?" — every component's index is present AND every shard it
    /// references exists, with no in-progress `*.incomplete` markers. (Just checking the small
    /// index files gives a false positive when a download was interrupted mid-shard.)
    public func isDownloaded(repoId: String) -> Bool {
        let fm = FileManager.default
        let root = localURL(repoId: repoId)
        guard fm.fileExists(atPath: root.path) else { return false }
        // Any in-progress download marker under the repo means it's incomplete.
        if let walker = fm.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let url as URL in walker where url.pathExtension == "incomplete" { return false }
        }
        for component in ["transformer", "text_encoder", "vae"] {
            let dir = root.appending(component: component)
            let index = dir.appending(component: "model.safetensors.index.json")
            guard let data = try? Data(contentsOf: index),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = json["weight_map"] as? [String: String] else { return false }
            let shards = Set(weightMap.values)
            guard !shards.isEmpty else { return false }
            for shard in shards where !fm.fileExists(atPath: dir.appending(component: shard).path) {
                return false
            }
        }
        return true
    }

    /// Download (idempotent) and return the local model directory. `progress` reports 0…1.
    /// `matching` optionally restricts which files are fetched (empty = the whole repo).
    @discardableResult
    public func download(repoId: String, matching globs: [String] = [],
                         progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        #if os(iOS)
        // On iOS a poisoned shared URLCache entry can replay a stale/empty repo file listing,
        // making snapshot() "succeed" while downloading nothing. Force fresh metadata.
        URLCache.shared.removeAllCachedResponses()
        #endif

        // An empty listing must never masquerade as a successful (empty) download — surface it.
        let listed = (try? await Hub.getFilenames(from: Hub.Repo(id: repoId), matching: globs)) ?? []
        guard !listed.isEmpty else { throw ModelDownloadError.emptyFileList(repoId) }

        let hub = HubApi(downloadBase: downloadBase)
        let url = try await hub.snapshot(from: repoId, matching: globs) { (p: Progress) in
            progress(p.fractionCompleted)
        }
        // A snapshot that returns "success" without materializing the weights (purged cache, no-op
        // transfer, partial) must surface as a retriable error — not a corrupt success that later
        // detonates as a missing-tensor error at generate time. Only meaningful for a full-repo fetch
        // (globs == []); a partial fetch wouldn't satisfy isDownloaded.
        if globs.isEmpty, !isDownloaded(repoId: repoId) {
            throw ModelDownloadError.incompleteDownload(repoId)
        }
        return url
    }
}

/// Errors surfaced by the in-app model downloader so silent/partial failures become visible + retriable.
public enum ModelDownloadError: LocalizedError {
    case emptyFileList(String)
    case incompleteDownload(String)
    public var errorDescription: String? {
        switch self {
        case .emptyFileList(let repo):
            return "Couldn’t list files for \(repo). Check your network connection and try again."
        case .incompleteDownload(let repo):
            return "Download didn’t finish for \(repo) — some weight files are missing. Tap download again to resume."
        }
    }
}
