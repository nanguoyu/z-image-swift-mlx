@preconcurrency import MLX
import MLXNN
import MLXFast
import Foundation

// AutoencoderKL VAE decoder (standard SD-style: 4 latent channels, 8x upsample, scale 0.18215).
// Works in NHWC (MLX conv layout) internally. Decoder only for now (text-to-image); the encoder
// (img2img) and exact weight loading are follow-ups. Structure + diffusers key names follow the
// reference; conv weights are PyTorch OIHW and need an O,H,W,I transpose at load (TODO). Numerics
// require GPU parity validation — nothing has run.

private func groupNorm(_ channels: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true)
}
private func conv3(_ inC: Int, _ outC: Int) -> Conv2d {
    Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 3, stride: 1, padding: 1)
}

/// ResNet block: GroupNorm→silu→Conv3 ×2 with an optional 1x1 shortcut. NHWC.
final class VAEResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(_ inC: Int, _ outC: Int) {
        self._norm1.wrappedValue = groupNorm(inC)
        self._conv1.wrappedValue = conv3(inC, outC)
        self._norm2.wrappedValue = groupNorm(outC)
        self._conv2.wrappedValue = conv3(outC, outC)
        self._convShortcut.wrappedValue = inC == outC
            ? nil
            : Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 1, stride: 1, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        return (convShortcut?(x) ?? x) + h
    }
}

/// Single-head spatial self-attention used in the VAE mid block. NHWC.
final class VAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNormLayer: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear   // reference key `to_out.0`

    private let channels: Int
    init(_ channels: Int) {
        self.channels = channels
        self._groupNormLayer.wrappedValue = groupNorm(channels)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = Linear(channels, channels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let normed = groupNormLayer(x).reshaped([b, h * w, c])
        let q = toQ(normed).reshaped([b, 1, h * w, c])
        let k = toK(normed).reshaped([b, 1, h * w, c])
        let v = toV(normed).reshaped([b, 1, h * w, c])
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(c)), mask: nil)
        let out = toOut(attn.reshaped([b, h * w, c])).reshaped([b, h, w, c])
        return x + out
    }
}

/// Nearest 2x upsample (NHWC) + Conv3.
final class VAEUpsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ channels: Int) {
        self._conv.wrappedValue = conv3(channels, channels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let up = broadcast(x.reshaped([b, h, 1, w, 1, c]), to: [b, h, 2, w, 2, c])
            .reshaped([b, h * 2, w * 2, c])
        return conv(up)
    }
}

/// A decoder up-stage: `count` resnets (first changes channels) + optional upsample.
final class VAEUpBlock: Module {
    let resnets: [VAEResnetBlock]
    @ModuleInfo(key: "upsamplers") var upsamplers: [VAEUpsample]
    init(_ inC: Int, _ outC: Int, count: Int, upsample: Bool) {
        self.resnets = (0..<count).map { VAEResnetBlock($0 == 0 ? inC : outC, outC) }
        self._upsamplers.wrappedValue = upsample ? [VAEUpsample(outC)] : []
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for u in upsamplers { h = u(h) }
        return h
    }
}

/// AutoencoderKL decoder. `decode` maps a latent `[B, 4, h, w]` (NCHW) → image `[B, H, W, 3]`
/// (NHWC) in [-1, 1].
public final class ZImageVAE: Module {
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    // decoder.* submodules
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_resnet1") var midResnet1: VAEResnetBlock
    @ModuleInfo(key: "mid_attn") var midAttn: VAEAttention
    @ModuleInfo(key: "mid_resnet2") var midResnet2: VAEResnetBlock
    let upBlocks: [VAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public override init() {
        let latent = ZImageConfig.VAE.latentChannels
        // reversed block channels [512, 512, 256, 128]
        let ch = ZImageConfig.VAE.blockChannels.reversed().map { $0 }
        let top = ch[0]
        self._postQuantConv.wrappedValue =
            Conv2d(inputChannels: latent, outputChannels: latent, kernelSize: 1, stride: 1, padding: 0)
        self._convIn.wrappedValue = conv3(latent, top)
        self._midResnet1.wrappedValue = VAEResnetBlock(top, top)
        self._midAttn.wrappedValue = VAEAttention(top)
        self._midResnet2.wrappedValue = VAEResnetBlock(top, top)
        var blocks: [VAEUpBlock] = []
        var inC = top
        for (i, outC) in ch.enumerated() {
            blocks.append(VAEUpBlock(inC, outC, count: 3, upsample: i < ch.count - 1))
            inC = outC
        }
        self.upBlocks = blocks
        self._convNormOut.wrappedValue = groupNorm(ch.last!)
        self._convOut.wrappedValue = conv3(ch.last!, 3)
        super.init()
    }

    public func decode(_ latentNCHW: MLXArray) -> MLXArray {
        let scaled = latentNCHW / ZImageConfig.VAE.scaleFactor
        var h = scaled.transposed(0, 2, 3, 1)        // NCHW -> NHWC
        h = postQuantConv(h)
        h = convIn(h)
        h = midResnet1(h); h = midAttn(h); h = midResnet2(h)
        for block in upBlocks { h = block(h) }
        h = convOut(silu(convNormOut(h)))            // [B, H, W, 3]
        return tanh(h)                                // -> [-1, 1]
    }
}
