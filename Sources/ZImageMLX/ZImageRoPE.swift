@preconcurrency import MLX
import Foundation

// Z-Image S3-DiT 3D-axes RoPE. Bit-for-bit with Tongyi-MAI/Z-Image src/zimage/transformer.py and
// the diffusers ZImageTransformer2DModel port (canonical math from Lumina-Image-2.0).
//
// head_dim 128 splits across three position axes (t, h, w) by axes_dims [32, 48, 48] → 16+24+24=64
// complex pairs. Rotation is INTERLEAVED (GPT-J / "traditional": adjacent dims 2k,2k+1 form a pair),
// NOT NeoX half-split. cos/sin are shared across all heads and applied identically to q and k (never
// v), strictly AFTER per-head QK-RMSNorm. Positions are per-token (t,h,w); caption tokens get
// t=1..L with h=w=0, image patch tokens get a constant t=L+1 with h=row, w=col.
enum ZImageRoPE {

    /// Inverse frequencies for one axis: `1 / theta^((2j)/d)`, j in 0..<d/2. The exponent
    /// denominator is the FULL `d` (arange(0,d,2)/d), per the reference. Returns `[d/2]` Float32.
    static func invFreq(d: Int, theta: Float) -> MLXArray {
        let half = d / 2
        return MLXArray((0..<half).map { 1.0 / pow(theta, Float(2 * $0) / Float(d)) })
    }

    /// Build the `[N, 64]` cos/sin tables from per-token positions `posT/posH/posW` (each `[N]`,
    /// Float32). Concatenation order is t ++ h ++ w (slots 0..15 t, 16..39 h, 40..63 w).
    static func tables(posT: MLXArray, posH: MLXArray, posW: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let theta = ZImageConfig.DiT.ropeTheta
        let dims = ZImageConfig.DiT.ropeAxesDims              // [32, 48, 48]
        let invT = invFreq(d: dims[0], theta: theta)         // [16]
        let invH = invFreq(d: dims[1], theta: theta)         // [24]
        let invW = invFreq(d: dims[2], theta: theta)         // [24]
        let angT = posT.reshaped([-1, 1]) * invT.reshaped([1, -1])   // [N, 16]
        let angH = posH.reshaped([-1, 1]) * invH.reshaped([1, -1])   // [N, 24]
        let angW = posW.reshaped([-1, 1]) * invW.reshaped([1, -1])   // [N, 24]
        let ang = concatenated([angT, angH, angW], axis: -1)         // [N, 64]
        return (cos(ang), sin(ang))
    }

    /// Apply interleaved rotation to `x` `[B, H, N, 128]` using cos/sin `[N, 64]`. Computed in
    /// Float32, cast back to `x.dtype`. For adjacent pair (a, b): out = (a·cos − b·sin, a·sin + b·cos).
    static func apply(_ x: MLXArray, cos cosTable: MLXArray, sin sinTable: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), n = x.dim(2), pairs = x.dim(3) / 2
        let xf = x.asType(.float32).reshaped([b, h, n, pairs, 2])
        let lanes = split(xf, parts: 2, axis: -1)
        let a = lanes[0].squeezed(axis: -1)                  // even lane [B,H,N,64]
        let bb = lanes[1].squeezed(axis: -1)                 // odd  lane [B,H,N,64]
        let c = cosTable.reshaped([1, 1, n, pairs])
        let s = sinTable.reshaped([1, 1, n, pairs])
        let outEven = a * c - bb * s
        let outOdd = a * s + bb * c
        return stacked([outEven, outOdd], axis: -1).reshaped([b, h, n, pairs * 2]).asType(x.dtype)
    }

    /// Per-token position ids for the image patch grid (`hp` rows × `wp` cols, raster h-outer /
    /// w-inner to match the patchify transpose) and the caption tokens (`length` L). Caption is
    /// numbered first on the t-axis (t=1..L, h=w=0); image patches share t=L+1 with h=row, w=col.
    /// Returns the per-segment position arrays so the refiners can build their own tables.
    static func positions(hp: Int, wp: Int, captionLength L: Int)
        -> (imgT: MLXArray, imgH: MLXArray, imgW: MLXArray, capT: MLXArray, capH: MLXArray, capW: MLXArray) {
        let hRange = MLXArray((0..<hp).map { Float($0) })
        let wRange = MLXArray((0..<wp).map { Float($0) })
        let imgH = broadcast(hRange.reshaped([hp, 1]), to: [hp, wp]).reshaped([hp * wp])   // h outer
        let imgW = broadcast(wRange.reshaped([1, wp]), to: [hp, wp]).reshaped([hp * wp])   // w inner
        let imgT = MLXArray(Array(repeating: Float(L + 1), count: hp * wp))
        let capT = MLXArray((0..<L).map { Float(1 + $0) })
        let capH = MLXArray(Array(repeating: Float(0), count: max(L, 1)))[0 ..< L]
        let capW = capH
        return (imgT, imgH, imgW, capT, capH, capW)
    }
}
