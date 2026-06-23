# z-image-swift-mlx

MLX/Swift implementation of [Z-Image](https://github.com/Tongyi-MAI/Z-Image) (Tongyi, Apache-2.0):
a 6B single-stream **S3-DiT** with a **Qwen3-4B** text encoder, distilled for 8-step generation.

Conforms to [`swift-diffusion-core`](../swift-diffusion-core)'s `DiffusionArchitecture`, so the
shared `MLXDiffusionEngine` can run it on macOS and iOS — including block-streaming partial
load for memory-constrained devices.

## Status

**Working** — runs end-to-end in pure Swift+MLX on the real 4-bit weights
([`deepsweet/Z-Image-Turbo-6B-MLX-Q4`](https://huggingface.co/deepsweet/Z-Image-Turbo-6B-MLX-Q4))
and generates coherent, prompt-faithful images (validated at 1024×1024, 8 steps). Implemented: the
Qwen3-4B encoder, the S3-DiT denoiser with real 3D-axes RoPE, the AutoencoderKL VAE, 4-bit weight
loading (keys verified against the checkpoint), and the flow-match denoise with Z-Image's timestep
(`1−σ`) and velocity-negation conventions. See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the
architecture map and the remaining polish/integration items.

## Usage

```swift
import ZImageMLX

let pipeline = ZImagePipeline(modelDirectory: URL(fileURLWithPath: "…/Z-Image-Turbo-6B-MLX-Q4"))
try await pipeline.loadModels()
let image = try pipeline.generate(prompt: "a red panda on a mossy rock", size: 1024, steps: 8, seed: 42)
```

The same `ZImageDenoiser` also conforms to `swift-diffusion-core`'s `Denoiser`, so it can be driven
by the block-streaming `MLXDiffusionEngine` (the intended iPhone partial-load path).

### CLI demo

```
swift run zimage-demo <model-dir> "<prompt>" [size=1024] [steps=8] [out.png]
```

Run via **Xcode** (it compiles MLX's Metal lib automatically). For `swift run` on Xcode 26, do the
two one-time fixes the CLI build doesn't: the target already adds `/usr/lib` to its rpath (resolves
`@rpath/libc++.1.dylib`); and `swift build` does **not** compile MLX's `default.metallib`, so build
once in Xcode and copy `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` to `mlx.metallib`
beside the `swift run` executable (`.build/<triple>/debug/mlx.metallib`).

## License

Apache-2.0. Model weights are Apache-2.0 (Tongyi-MAI).
