import CoreGraphics
import DiffusionCore
import Foundation

/// A `DiffusionEngine` facade over `ZImagePipeline`, so the app drives Z-Image through the same
/// boundary as FLUX (`Flux2FacadeEngine`). This is the resident macOS path; the block-streaming
/// `MLXDiffusionEngine` + `ZImageArchitecture` is the (future) iPhone partial-load path.
///
/// `source` is intentionally ignored — `ZImagePipeline` reads the three component folders
/// (`text_encoder/`, `transformer/`, `vae/`, `tokenizer/`) from `modelDirectory`, because the
/// seam's single flat `WeightSource` can't represent three colliding key namespaces.
public actor ZImageFacadeEngine: DiffusionEngine {
    private let modelDirectory: URL
    private var pipeline: ZImagePipeline?

    /// `modelDirectory` is the downloaded model folder (e.g. a `Z-Image-Turbo-6B-MLX-Q4` checkout).
    public init(modelDirectory: URL) { self.modelDirectory = modelDirectory }

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        // Resident facade: estimate runtime peak above on-disk size (weights + working buffers)
        // and gate against the device's memory budget on BOTH Mac and phone.
        let estimatedPeak = variant.approximateBytes + variant.approximateBytes / 3
        let fits = estimatedPeak < device.memoryBudgetBytes
        if device.isPhone {
            return EngineCapabilities(
                runnable: fits, residency: fits ? .resident : .unsupported,
                estimatedPeakBytes: estimatedPeak,
                note: fits ? "Resident on a large-RAM iPhone" : "Use the streaming engine")
        }
        return EngineCapabilities(runnable: fits, residency: fits ? .resident : .unsupported,
                                  estimatedPeakBytes: estimatedPeak,
                                  note: fits ? "Runs on Mac" : "Insufficient memory")
    }

    public func load(_ model: DiffusionModel, variant: ModelVariant, source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        let pipeline = ZImagePipeline(modelDirectory: modelDirectory)
        try await pipeline.loadModels(progress: progress)
        self.pipeline = pipeline
    }

    public func generate(_ request: GenerationRequest,
                         progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage {
        guard let pipeline else { throw ZImageEngineError.notLoaded }
        // Surface the current limitations explicitly rather than silently dropping the request:
        // ZImagePipeline is text-to-image and square-only for now.
        if request.referenceImage != nil { throw ZImageEngineError.imageToImageUnsupported }
        guard request.size.width == request.size.height else { throw ZImageEngineError.nonSquareUnsupported }
        progress(.preparing)
        let image = try pipeline.generate(
            prompt: request.prompt, size: request.size.width, steps: request.steps, seed: request.seed
        ) { step, total in
            progress(.denoising(step: step, total: total, preview: nil))
        }
        progress(.finished(image))
        return image
    }

    public func unload() async {
        pipeline?.unload()
        pipeline = nil
    }
}

public enum ZImageEngineError: Error, CustomStringConvertible {
    case notLoaded
    case imageToImageUnsupported
    case nonSquareUnsupported
    public var description: String {
        switch self {
        case .notLoaded: return "ZImageFacadeEngine: call load(...) before generate(...)"
        case .imageToImageUnsupported: return "ZImageFacadeEngine: image-to-image is not supported yet (text-to-image only)"
        case .nonSquareUnsupported: return "ZImageFacadeEngine: non-square sizes are not supported yet"
        }
    }
}
