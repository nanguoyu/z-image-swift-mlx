# z-image-swift-mlx

MLX/Swift implementation of [Z-Image](https://github.com/Tongyi-MAI/Z-Image) (Tongyi, Apache-2.0):
a 6B single-stream **S3-DiT** with a **Qwen3-4B** text encoder, distilled for 8-step generation.

Conforms to [`swift-diffusion-core`](../swift-diffusion-core)'s `DiffusionArchitecture`, so the
shared `MLXDiffusionEngine` can run it on macOS and iOS — including block-streaming partial
load for memory-constrained devices.

## Status

Scaffold — `ZImageArchitecture` declares its `spec` and the seam methods; the S3-DiT blocks,
Qwen3-4B encoding, and VAE land in Phase 0 (the first non-FLUX model, used to measure the real
per-architecture Swift cost).

## License

Apache-2.0. Model weights are Apache-2.0 (Tongyi-MAI).
