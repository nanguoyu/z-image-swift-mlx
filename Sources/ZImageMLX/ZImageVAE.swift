@preconcurrency import MLX
import MLXNN
import MLXFast
import Foundation

// AutoencoderKL VAE (diffusers layout) for Z-Image. Module tree + weight keys match the reference
// checkpoint exactly (encoder.* + decoder.*, no quant_conv/post_quant_conv — this checkpoint ships
// none). Works in NHWC (MLX conv layout) internally; conv weights are PyTorch OIHW and get an
// O,H,W,I transpose at load (see ZImageWeights.conv2dWeight).
//
// NOTE: the converter wraps the top convs/norm asymmetrically — decoder uses `conv_in.conv`,
// encoder uses `conv_in.conv2d`, and both norm-outs use `.norm`. The thin wrappers below reproduce
// those exact keys. Forward numerics (scale factor, final activation, attention) still need GPU
// parity validation.

private func groupNorm(_ channels: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true)
}
private func conv3(_ inC: Int, _ outC: Int, stride: Int = 1) -> Conv2d {
    Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 3,
           stride: IntOrPair((stride, stride)), padding: 1)
}
private func conv1(_ inC: Int, _ outC: Int) -> Conv2d {
    Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 1, stride: 1, padding: 0)
}

/// Decoder top-conv wrapper: child Conv2d keyed `conv` (e.g. `decoder.conv_in.conv`).
final class ConvWrap: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ c: Conv2d) { self._conv.wrappedValue = c; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}
/// Encoder top-conv wrapper: child Conv2d keyed `conv2d` (e.g. `encoder.conv_in.conv2d`).
final class Conv2dWrap: Module {
    @ModuleInfo(key: "conv2d") var conv2d: Conv2d
    init(_ c: Conv2d) { self._conv2d.wrappedValue = c; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2d(x) }
}
/// Norm-out wrapper: child GroupNorm keyed `norm` (e.g. `decoder.conv_norm_out.norm`).
final class NormWrap: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    init(_ n: GroupNorm) { self._norm.wrappedValue = n; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x) }
}

/// ResNet block: GroupNorm→silu→Conv3 ×2 with an optional 1×1 shortcut. NHWC.
final class VAEResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1Layer: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2Layer: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(_ inC: Int, _ outC: Int) {
        self._norm1.wrappedValue = groupNorm(inC)
        self._conv1Layer.wrappedValue = conv3(inC, outC)
        self._norm2.wrappedValue = groupNorm(outC)
        self._conv2Layer.wrappedValue = conv3(outC, outC)
        self._convShortcut.wrappedValue = inC == outC ? nil : conv1(inC, outC)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1Layer(silu(norm1(x)))
        h = conv2Layer(silu(norm2(h)))
        return (convShortcut?(x) ?? x) + h
    }
}

/// Single-head spatial self-attention used in the VAE mid block. NHWC. `to_out` is a 1-element
/// list so its key is `to_out.0`.
final class VAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNormLayer: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    init(_ channels: Int) {
        self._groupNormLayer.wrappedValue = groupNorm(channels)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = [Linear(channels, channels)]
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
        let out = toOut[0](attn.reshaped([b, h * w, c])).reshaped([b, h, w, c])
        return x + out
    }
}

/// Mid block: resnet → attention → resnet. Keys: `resnets.{0,1}`, `attentions.0`.
final class VAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock]
    @ModuleInfo(key: "attentions") var attentions: [VAEAttention]
    init(_ channels: Int) {
        self._resnets.wrappedValue = [VAEResnetBlock(channels, channels), VAEResnetBlock(channels, channels)]
        self._attentions.wrappedValue = [VAEAttention(channels)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x)
        h = attentions[0](h)
        h = resnets[1](h)
        return h
    }
}

/// Nearest 2× upsample (NHWC) + Conv3. Key: `conv`.
final class VAEUpsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ channels: Int) { self._conv.wrappedValue = conv3(channels, channels); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let up = broadcast(x.reshaped([b, h, 1, w, 1, c]), to: [b, h, 2, w, 2, c])
            .reshaped([b, h * 2, w * 2, c])
        return conv(up)
    }
}

/// Strided-conv 2× downsample (NHWC). Key: `conv`.
final class VAEDownsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ channels: Int) {
        // diffusers pads (0,1,0,1) then stride-2 valid conv; here a stride-2 padded conv approximates it.
        self._conv.wrappedValue = conv3(channels, channels, stride: 2)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// Decoder up-stage: `count` resnets (first changes channels) + optional upsample.
final class VAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock]
    @ModuleInfo(key: "upsamplers") var upsamplers: [VAEUpsample]
    init(_ inC: Int, _ outC: Int, count: Int, upsample: Bool) {
        self._resnets.wrappedValue = (0..<count).map { VAEResnetBlock($0 == 0 ? inC : outC, outC) }
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

/// Encoder down-stage: `count` resnets (first changes channels) + optional downsample.
final class VAEDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock]
    @ModuleInfo(key: "downsamplers") var downsamplers: [VAEDownsample]
    init(_ inC: Int, _ outC: Int, count: Int, downsample: Bool) {
        self._resnets.wrappedValue = (0..<count).map { VAEResnetBlock($0 == 0 ? inC : outC, outC) }
        self._downsamplers.wrappedValue = downsample ? [VAEDownsample(outC)] : []
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        for d in downsamplers { h = d(h) }
        return h
    }
}

/// Decoder: latent `[B,4,h,w]` (NCHW) → image `[B,H,W,3]` (NHWC).
final class VAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: ConvWrap
    @ModuleInfo(key: "mid_block") var midBlock: VAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [VAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: NormWrap
    @ModuleInfo(key: "conv_out") var convOut: ConvWrap

    override init() {
        let latent = ZImageConfig.VAE.latentChannels
        let ch = ZImageConfig.VAE.blockChannels.reversed().map { $0 }   // [512, 512, 256, 128]
        let top = ch[0]
        self._convIn.wrappedValue = ConvWrap(conv3(latent, top))
        self._midBlock.wrappedValue = VAEMidBlock(top)
        var blocks: [VAEUpBlock] = []
        var inC = top
        for (i, outC) in ch.enumerated() {
            blocks.append(VAEUpBlock(inC, outC, count: 3, upsample: i < ch.count - 1))
            inC = outC
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = NormWrap(groupNorm(ch.last!))
        self._convOut.wrappedValue = ConvWrap(conv3(ch.last!, 3))
        super.init()
    }
    func callAsFunction(_ latentNHWC: MLXArray) -> MLXArray {
        var h = convIn(latentNHWC)
        h = midBlock(h)
        for block in upBlocks { h = block(h) }
        return convOut(silu(convNormOut(h)))
    }
}

/// Encoder: image `[B,H,W,3]` (NHWC) → moments `[B,h,w,8]` (mean ‖ logvar).
final class VAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2dWrap
    @ModuleInfo(key: "down_blocks") var downBlocks: [VAEDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: VAEMidBlock
    @ModuleInfo(key: "conv_norm_out") var convNormOut: NormWrap
    @ModuleInfo(key: "conv_out") var convOut: Conv2dWrap

    override init() {
        let latent = ZImageConfig.VAE.latentChannels
        let ch = ZImageConfig.VAE.blockChannels   // [128, 256, 512, 512]
        self._convIn.wrappedValue = Conv2dWrap(conv3(3, ch[0]))
        var blocks: [VAEDownBlock] = []
        var inC = ch[0]
        for (i, outC) in ch.enumerated() {
            blocks.append(VAEDownBlock(inC, outC, count: 2, downsample: i < ch.count - 1))
            inC = outC
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = VAEMidBlock(ch.last!)
        self._convNormOut.wrappedValue = NormWrap(groupNorm(ch.last!))
        self._convOut.wrappedValue = Conv2dWrap(conv3(ch.last!, 2 * latent))
        super.init()
    }
    func callAsFunction(_ imageNHWC: MLXArray) -> MLXArray {
        var h = convIn(imageNHWC)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        return convOut(silu(convNormOut(h)))
    }
}

/// AutoencoderKL. `decode` maps a latent `[B,4,h,w]` (NCHW) → image `[B,H,W,3]` (NHWC); `encode`
/// maps an image `[B,3,H,W]` (NCHW) → latent `[B,4,h,w]` (NCHW). Numerics need parity validation.
public final class ZImageVAE: Module {
    @ModuleInfo(key: "decoder") var decoder: VAEDecoder
    @ModuleInfo(key: "encoder") var encoder: VAEEncoder

    public override init() {
        self._decoder.wrappedValue = VAEDecoder()
        self._encoder.wrappedValue = VAEEncoder()
        super.init()
    }

    public func decode(_ latentNCHW: MLXArray) -> MLXArray {
        let scaled = latentNCHW / ZImageConfig.VAE.scaleFactor
        let nhwc = scaled.transposed(0, 2, 3, 1)        // NCHW -> NHWC
        return decoder(nhwc)                            // [B, H, W, 3]
    }

    /// Encode to the distribution mean (deterministic), scaled to latent space. NCHW in/out.
    public func encode(_ imageNCHW: MLXArray) -> MLXArray {
        let nhwc = imageNCHW.transposed(0, 2, 3, 1)     // NCHW -> NHWC
        let moments = encoder(nhwc)                     // [B, h, w, 8]
        let c = ZImageConfig.VAE.latentChannels
        let mean = moments[.ellipsis, 0 ..< c]          // mean half
        return mean.transposed(0, 3, 1, 2) * ZImageConfig.VAE.scaleFactor   // NHWC -> NCHW
    }
}
