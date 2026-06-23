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

    /// Cheap "already downloaded?" check: all three component indices are present on disk.
    public func isDownloaded(repoId: String) -> Bool {
        let root = localURL(repoId: repoId)
        return ["transformer", "text_encoder", "vae"].allSatisfy {
            FileManager.default.fileExists(
                atPath: root.appending(component: $0).appending(component: "model.safetensors.index.json").path)
        }
    }

    /// Download (idempotent) and return the local model directory. `progress` reports 0…1.
    /// `matching` optionally restricts which files are fetched (empty = the whole repo).
    @discardableResult
    public func download(repoId: String, matching globs: [String] = [],
                         progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let hub = HubApi(downloadBase: downloadBase)
        return try await hub.snapshot(from: repoId, matching: globs) { (p: Progress) in
            progress(p.fractionCompleted)
        }
    }
}
