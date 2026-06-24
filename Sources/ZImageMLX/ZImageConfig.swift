import Foundation

/// Architecture constants for Z-Image, from the reference modeling code + HF configs.
/// See IMPLEMENTATION.md for sources. These are exact; the forward passes built on them still
/// require numerical parity validation against the Python reference on a GPU.
public enum ZImageConfig {

    /// S3-DiT single-stream transformer.
    public enum DiT {
        public static let dim = 3840
        public static let heads = 30
        public static let headDim = 128            // dim / heads
        public static let layers = 30              // main blocks
        public static let noiseRefiners = 2
        public static let contextRefiners = 2
        public static let patchSize = 2
        public static let vaeLatentChannels = 16   // transformer in_channels (FLUX VAE latent)
        public static let patchedChannels = 64     // vaeLatentChannels * patchSize^2 = 16 * 4
        public static let ffnHidden = 10240        // int(dim / 3 * 8)
        public static let rmsEps: Float = 1e-5     // norm_eps
        public static let captionDim = 2560        // text-encoder hidden (cap_feat_dim), projected to `dim`
        public static let adaLNInputDim = 256       // ADALN_EMBED_DIM: t_embedder output → adaLN input
        public static let tEmbedderHidden = 1024    // t_embedder MLP hidden (256 → 1024 → 256)
        public static let ropeTheta: Float = 256
        public static let ropeAxesDims = [32, 48, 48]
        public static let ropeAxesLens = [1536, 512, 512]
        public static let tScale: Float = 1000.0   // timestep is scaled by 1000 before embedding
    }

    /// Qwen3-4B text encoder (reimplemented in MLX; layer[-2] hidden state is used).
    public enum TextEncoder {
        public static let layers = 36
        public static let hidden = 2560
        public static let heads = 32
        public static let kvHeads = 8
        public static let headDim = 128
        public static let ffnHidden = 9728
        public static let rmsEps: Float = 1e-6
        public static let ropeTheta: Float = 1_000_000
        public static let extractLayerFromEnd = 2  // second-to-last
        public static let maxSequenceLength = 512
    }

    /// AutoencoderKL VAE (FLUX-family: 16 latent channels, scale 0.3611, shift 0.1159).
    public enum VAE {
        public static let latentChannels = 16
        public static let downsampleFactor = 8
        public static let scaleFactor: Float = 0.3611
        public static let shiftFactor: Float = 0.1159
        public static let blockChannels = [128, 256, 512, 512]
    }

    /// Flow-match Euler scheduler. Matches mflux's Z-Image schedule: exponential time-shift with
    /// mu = 1 (so `shift = exp(1) ≈ 2.71828`, algebraically the same as the linear form) and a
    /// `shift_terminal` of 0.02 so the last working sigma is 0.02 (the model isn't evaluated below
    /// its calibrated range — going to ~0.003 produced grain).
    public enum Scheduler {
        public static let shift: Float = 2.7182817   // exp(1)
        public static let shiftTerminal: Float = 0.02
        public static let defaultSteps = 8
    }
}
