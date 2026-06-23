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
        // Resident facade: runnable when the weights fit the device's working-set budget.
        let fits = variant.approximateBytes < device.memoryBudgetBytes
        if device.isPhone {
            return EngineCapabilities(
                runnable: fits, residency: fits ? .resident : .unsupported,
                estimatedPeakBytes: variant.approximateBytes,
                note: fits ? "Resident on a large-RAM iPhone" : "Use the streaming engine")
        }
        return EngineCapabilities(runnable: true, residency: .resident,
                                  estimatedPeakBytes: variant.approximateBytes, note: "Runs on Mac")
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
        progress(.preparing)
        // NOTE: img2img (request.referenceImage) is not yet supported by the facade — text-to-image
        // only for now. Square `size.width` is used (the catalog ships square sizes).
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
    public var description: String {
        switch self { case .notLoaded: return "ZImageFacadeEngine: call load(...) before generate(...)" }
    }
}
