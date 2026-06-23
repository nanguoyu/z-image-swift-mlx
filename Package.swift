// swift-tools-version: 5.10
import PackageDescription

// z-image-swift-mlx — MLX/Swift implementation of Z-Image (Tongyi): single-stream S3-DiT +
// Qwen3-4B text encoder. Conforms to swift-diffusion-core's `DiffusionArchitecture` so the
// shared engine can drive it (including block-streaming partial load).
//
// Its own public repo (nanguoyu/z-image-swift-mlx); depends on swift-diffusion-core by URL.
let package = Package(
    name: "z-image-swift-mlx",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ZImageMLX", targets: ["ZImageMLX"]),
        .executable(name: "zimage-demo", targets: ["zimage-demo"]),
    ],
    dependencies: [
        .package(url: "https://github.com/nanguoyu/swift-diffusion-core", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        // Qwen3-4B tokenizer / text encoder support. NOTE: verify the product name + pin in Phase 0.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "ZImageMLX",
            dependencies: [
                .product(name: "DiffusionCore", package: "swift-diffusion-core"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        // Tiny CLI: `swift run zimage-demo <model-dir> "<prompt>" [size] [steps] [out.png]`.
        // For `swift run` on Xcode 26, copy an Xcode-built MLX `default.metallib` to
        // `mlx.metallib` beside the executable (see README); Xcode runs need no extra steps.
        .executableTarget(
            name: "zimage-demo",
            dependencies: ["ZImageMLX"],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "/usr/lib"])]
        ),
    ]
)
