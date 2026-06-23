import Foundation
import MLX
import ZImageMLX
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

// swift run zimage-demo <model-dir> "<prompt>" [size] [steps] [out.png]
//   or: swift run zimage-demo probe <repoId>     (download only *.json — a cheap download audit)
setvbuf(stdout, nil, _IONBF, 0)
let args = CommandLine.arguments

if args.count >= 3, args[1] == "probe" {
    let repoId = args[2]
    let base = URL(fileURLWithPath: NSTemporaryDirectory()).appending(component: "zimage-probe")
    let downloader = ModelDownloader(downloadBase: base)
    print("probing \(repoId) → \(base.path) (json only)")
    let url = try await downloader.download(repoId: repoId, matching: ["**/*.json"]) { f in
        print(String(format: "  %.0f%%", f * 100))
    }
    let files = (try? FileManager.default.subpathsOfDirectory(atPath: url.path)) ?? []
    print("resolved dir: \(url.path)")
    print("files: \(files.filter { $0.hasSuffix(".json") }.sorted().joined(separator: ", "))")
    print("localURL matches resolved: \(downloader.localURL(repoId: repoId).path == url.path)")
    exit(0)
}

guard args.count >= 3 else {
    print("usage: zimage-demo <model-dir> \"<prompt>\" [size=1024] [steps=8] [out=zimage.png]")
    print("   or: zimage-demo probe <repoId>")
    exit(2)
}
let modelDir = URL(fileURLWithPath: args[1])
let prompt = args[2]
let size = args.count > 3 ? (Int(args[3]) ?? 1024) : 1024
let steps = args.count > 4 ? (Int(args[4]) ?? 8) : 8
let outURL = URL(fileURLWithPath: args.count > 5 ? args[5] : "zimage.png")

MLX.GPU.set(cacheLimit: 1 << 30)
let start = Date()
let pipeline = ZImagePipeline(modelDirectory: modelDir)
print("loading models from \(modelDir.path) …")
try await pipeline.loadModels { f in print(String(format: "  load %.0f%%", f * 100)) }
print(String(format: "loaded in %.1fs; generating \"%@\" @ %dx%d, %d steps", -start.timeIntervalSinceNow, prompt, size, size, steps))

let image = try pipeline.generate(prompt: prompt, size: size, steps: steps, seed: 42) { step, total in
    print("  step \(step)/\(total)")
}
if let dst = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.png.identifier as CFString, 1, nil) {
    CGImageDestinationAddImage(dst, image, nil)
    CGImageDestinationFinalize(dst)
    print(String(format: "wrote %@ (%dx%d) in %.1fs", outURL.path, image.width, image.height, -start.timeIntervalSinceNow))
}
