import Foundation
import Hub
import CryptoKit

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
    private struct HubFile: Sendable {
        var path: String
        var size: Int64?
        var sha256: String?
    }

    private struct DownloadManifest: Codable {
        struct File: Codable {
            var path: String
            var size: Int64?
            var sha256: String?
        }
        var version: Int = 1
        var files: [File]
    }

    private static let manifestFilename = ".mobile-diffuser-download-manifest.json"

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
            for case let url as URL in walker where url.pathExtension == "incomplete" || url.pathExtension == "part" { return false }
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
        return Self.verifyManifestIfPresent(at: root)
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

        if globs.isEmpty {
            let root = localURL(repoId: repoId)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let allFiles = try await fetchHubFiles(repoId: repoId)
            let files = allFiles.filter(Self.isModelFile)
            guard !files.isEmpty else { throw ModelDownloadError.emptyFileList(repoId) }
            let totalBytes = max(1, files.reduce(Int64(0)) { $0 + ($1.size ?? 0) })
            var completedBytes: Int64 = 0
            let session = Self.makeSession()
            for file in files {
                let destination = root.appendingPathComponent(file.path)
                if Self.fileMatches(destination, expectedSize: file.size, expectedSHA256: file.sha256) {
                    completedBytes += file.size ?? Self.fileSize(destination)
                    progress(min(1, Double(completedBytes) / Double(totalBytes)))
                    continue
                }
                let baseBytes = completedBytes
                try await downloadFile(repoId: repoId, file: file, to: destination, session: session) { bytes in
                    progress(min(1, Double(baseBytes + bytes) / Double(totalBytes)))
                }
                completedBytes += Self.fileSize(destination)
            }
            try Self.writeManifest(files, at: root)
            guard isDownloaded(repoId: repoId) else { throw ModelDownloadError.incompleteDownload(repoId) }
            progress(1)
            return root
        }

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

    private static func isModelFile(_ file: HubFile) -> Bool {
        file.path.hasSuffix(".safetensors") || file.path.hasSuffix(".json") || file.path == "tokenizer.model"
    }

    private static func makeSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }

    private func fetchHubFiles(repoId: String) async throws -> [HubFile] {
        let urlString = "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=1&expand=1"
        guard let url = URL(string: urlString) else { throw ModelDownloadError.invalidURL(urlString) }
        let (data, response) = try await Self.makeSession().data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw ModelDownloadError.emptyFileList(repoId)
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw ModelDownloadError.emptyFileList(repoId)
        }
        return json.compactMap { item in
            guard (item["type"] as? String) == "file", let path = item["path"] as? String else { return nil }
            let size = (item["size"] as? NSNumber)?.int64Value
            let lfs = item["lfs"] as? [String: Any]
            let sha256 = lfs?["oid"] as? String
            let lfsSize = (lfs?["size"] as? NSNumber)?.int64Value
            return HubFile(path: path, size: lfsSize ?? size, sha256: sha256)
        }
    }

    private func downloadFile(repoId: String, file: HubFile, to destination: URL, session: URLSession,
                              progress: @escaping @Sendable (Int64) -> Void) async throws {
        if Self.fileMatches(destination, expectedSize: file.size, expectedSHA256: file.sha256) {
            progress(file.size ?? Self.fileSize(destination)); return
        }
        let urlString = "https://huggingface.co/\(repoId)/resolve/main/\(file.path)"
        guard let url = URL(string: urlString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? urlString) else {
            throw ModelDownloadError.invalidURL(urlString)
        }
        var request = URLRequest(url: url)
        let partURL = destination.appendingPathExtension("part")
        var existingBytes = Self.fileSize(partURL)
        if existingBytes > 0 { request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range") }
        // Native chunked download to a temp file (URLSession.bytes' per-UInt8 AsyncSequence is far too
        // slow for multi-GB shards). Resume is handled via the Range header + the .part file.
        let (tempURL, response) = try await session.download(for: request)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        guard let http = response as? HTTPURLResponse else { throw ModelDownloadError.incompleteDownload(repoId) }
        if existingBytes > 0, http.statusCode != 206 {
            // Server ignored the Range and sent the full file — discard the partial to avoid corruption.
            try? FileManager.default.removeItem(at: partURL)
            existingBytes = 0
        }
        guard http.statusCode == 200 || http.statusCode == 206 else { throw ModelDownloadError.incompleteDownload(repoId) }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if existingBytes == 0 {
            if FileManager.default.fileExists(atPath: partURL.path) { try FileManager.default.removeItem(at: partURL) }
            try FileManager.default.moveItem(at: tempURL, to: partURL)
        } else {
            let outHandle = try FileHandle(forWritingTo: partURL)
            let inHandle = try FileHandle(forReadingFrom: tempURL)
            defer { try? outHandle.close(); try? inHandle.close() }
            try outHandle.seekToEnd()
            while let chunk = try inHandle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
                try outHandle.write(contentsOf: chunk)
            }
        }
        progress(Self.fileSize(partURL))
        guard Self.fileMatches(partURL, expectedSize: file.size, expectedSHA256: file.sha256) else {
            // Remove the corrupt partial so a retry re-downloads from scratch, not the same bad prefix.
            try? FileManager.default.removeItem(at: partURL)
            throw ModelDownloadError.hashMismatch(file.path)
        }
        if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
        try FileManager.default.moveItem(at: partURL, to: destination)
    }

    private static func writeManifest(_ files: [HubFile], at directory: URL) throws {
        let manifest = DownloadManifest(files: files.map { DownloadManifest.File(path: $0.path, size: $0.size, sha256: $0.sha256) })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: directory.appendingPathComponent(manifestFilename), options: .atomic)
    }

    private static func verifyManifestIfPresent(at directory: URL, verifyHashes: Bool = false) -> Bool {
        let url = directory.appendingPathComponent(manifestFilename)
        // No manifest (older app version, mflux, glob-fetched repo, HF CLI cache) or an unreadable one
        // is not evidence of a bad download — the caller already verified the shard structure. Don't
        // force a multi-GB re-download; only a readable manifest whose listed files fail means incomplete.
        guard FileManager.default.fileExists(atPath: url.path) else { return true }
        guard let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(DownloadManifest.self, from: data) else { return true }
        for file in manifest.files {
            guard fileMatches(directory.appendingPathComponent(file.path), expectedSize: file.size, expectedSHA256: file.sha256, verifyHash: verifyHashes) else {
                return false
            }
        }
        return true
    }

    private static func fileMatches(_ url: URL, expectedSize: Int64?, expectedSHA256: String?, verifyHash: Bool = true) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        if let expectedSize, fileSize(url) != expectedSize { return false }
        if verifyHash, let expectedSHA256, !expectedSHA256.isEmpty {
            guard (try? sha256Hex(of: url)) == expectedSHA256.lowercased() else { return false }
        }
        return true
    }

    private static func fileSize(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
    }

    private static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Errors surfaced by the in-app model downloader so silent/partial failures become visible + retriable.
public enum ModelDownloadError: LocalizedError {
    case emptyFileList(String)
    case incompleteDownload(String)
    case invalidURL(String)
    case hashMismatch(String)
    public var errorDescription: String? {
        switch self {
        case .emptyFileList(let repo):
            return "Couldn’t list files for \(repo). Check your network connection and try again."
        case .incompleteDownload(let repo):
            return "Download didn’t finish for \(repo) — some weight files are missing. Tap download again to resume."
        case .invalidURL(let url):
            return "Invalid download URL: \(url)"
        case .hashMismatch(let file):
            return "Hash verification failed for \(file). Tap download again to retry."
        }
    }
}
