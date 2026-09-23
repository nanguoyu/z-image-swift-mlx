import XCTest
@testable import ZImageMLX

/// The per-file install state behind `download(repoId:files:)`: what counts as downloaded, and what a
/// delete leaves behind. (The transfer itself is the same resumable path the whole-repo download uses.)
final class DownloaderFileTests: XCTestCase {
    private var base: URL!
    private let repo = "someone/model-GGUF"

    override func setUpWithError() throws {
        base = FileManager.default.temporaryDirectory.appendingPathComponent("dl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    func testRecordedFileOfTheRightSizeIsDownloaded() throws {
        let downloader = ModelDownloader(downloadBase: base)
        try place("model-Q4_K_M.gguf", bytes: 10)
        try record(["model-Q4_K_M.gguf": 10])
        XCTAssertTrue(downloader.isDownloaded(repoId: repo, files: ["model-Q4_K_M.gguf"]))
    }

    func testFileWithoutARecordIsNotDownloaded() throws {
        let downloader = ModelDownloader(downloadBase: base)
        try place("model-Q4_K_M.gguf", bytes: 10)
        XCTAssertFalse(downloader.isDownloaded(repoId: repo, files: ["model-Q4_K_M.gguf"]))
    }

    func testWrongSizeOrPartialIsNotDownloaded() throws {
        let downloader = ModelDownloader(downloadBase: base)
        try place("a.gguf", bytes: 9)
        try place("b.gguf", bytes: 10)
        try place("b.gguf.part", bytes: 3)
        try record(["a.gguf": 10, "b.gguf": 10])
        XCTAssertFalse(downloader.isDownloaded(repoId: repo, files: ["a.gguf"]))
        XCTAssertFalse(downloader.isDownloaded(repoId: repo, files: ["b.gguf"]), "a re-download is in progress")
    }

    func testEveryNamedFileMustBePresent() throws {
        let downloader = ModelDownloader(downloadBase: base)
        try place("a.gguf", bytes: 4)
        try record(["a.gguf": 4, "b.gguf": 4])
        XCTAssertTrue(downloader.isDownloaded(repoId: repo, files: ["a.gguf"]))
        XCTAssertFalse(downloader.isDownloaded(repoId: repo, files: ["a.gguf", "b.gguf"]))
    }

    func testDeleteRemovesTheFileItsPartialAndItsRecordOnly() throws {
        let downloader = ModelDownloader(downloadBase: base)
        try place("a.gguf", bytes: 4)
        try place("a.gguf.part", bytes: 2)
        try place("b.gguf", bytes: 4)
        try record(["a.gguf": 4, "b.gguf": 4])

        try downloader.delete(repoId: repo, files: ["a.gguf"])

        let root = downloader.localURL(repoId: repo)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("a.gguf").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("a.gguf.part").path))
        XCTAssertFalse(downloader.isDownloaded(repoId: repo, files: ["a.gguf"]))
        XCTAssertTrue(downloader.isDownloaded(repoId: repo, files: ["b.gguf"]), "other files stay installed")
    }

    func testFilesInSubdirectoriesUseTheirRepoPath() throws {
        let downloader = ModelDownloader(downloadBase: base)
        try place("vae/model_vae.safetensors", bytes: 6)
        try record(["vae/model_vae.safetensors": 6])
        XCTAssertTrue(downloader.isDownloaded(repoId: repo, files: ["vae/model_vae.safetensors"]))
    }

    // MARK: - Helpers

    private func place(_ path: String, bytes: Int) throws {
        let url = ModelDownloader(downloadBase: base).localURL(repoId: repo).appendingPathComponent(path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 7, count: bytes).write(to: url)
    }

    /// Writes the manifest a completed per-file download leaves behind.
    private func record(_ sizes: [String: Int]) throws {
        let root = ModelDownloader(downloadBase: base).localURL(repoId: repo)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let files = sizes.keys.sorted().map { ["path": $0, "size": sizes[$0]!] as [String: Any] }
        let json = try JSONSerialization.data(withJSONObject: ["version": 1, "files": files])
        try json.write(to: root.appendingPathComponent(".mobile-diffuser-download-manifest.json"))
    }
}
