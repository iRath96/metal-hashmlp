import Foundation
import Metal

public class AdamOptimizer {
    public struct Configuration {
        public let parameterCount: Int
        public let precision: FloatPrecision
        public let learningRate: Float
        public let weightDecay: Float /// L2 regularization
        
        public init(parameterCount: Int, precision: FloatPrecision, learningRate: Float, weightDecay: Float) {
            self.parameterCount = parameterCount
            self.precision = precision
            self.learningRate = learningRate
            self.weightDecay = weightDecay
        }
    }
    
    private struct Pipelines {
        let copyWeights: MTLComputePipelineState
        let performStep: MTLComputePipelineState
    }

    private struct Buffers {
        let weightsFull: MTLBuffer
        let firstMoment: MTLBuffer
        let secondMoment: MTLBuffer
    }
    
    public let weights: MTLBuffer
    public let config: Configuration
    public var iteration: Int = 0
    private let pipelines: Pipelines
    private let buffers: Buffers
    
    public init(for weights: MTLBuffer, with config: Configuration) throws {
        let device = weights.device
        self.weights = weights
        self.config = config
        
        let kernelConfig = """
using FloatMLP = \(config.precision.size == 2 ? "half" : "float");
using FloatFP = float;

"""
        
        let kernelPath = Bundle.module.url(forResource: "adam", withExtension: "metal", subdirectory: "metal")!
        let kernelSource = try! String(contentsOf: kernelPath)
        let library = try! device.makeLibrary(
            source: kernelConfig + kernelSource,
            options: MTLCompileOptions())
        
        let size = config.parameterCount * MemoryLayout<Float>.size
        self.buffers = .init(
            weightsFull: device.makeBuffer(named: "Adam Weights", length: size)!,
            firstMoment: device.makeBuffer(named: "Adam First Moment", length: size)!,
            secondMoment: device.makeBuffer(named: "Adam Second Moment", length: size)!)
        
        self.pipelines = .init(
            copyWeights: try library.makePipeline(named: "copy"),
            performStep: try library.makePipeline(named: "step"))
        
        let buffer = device.makeCommandQueue()!.makeCommandBuffer()!
        copyWeights(on: buffer)
        resetStatistics(on: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
    }
    
    private func copyWeights(on buffer: MTLCommandBuffer) {
        let encoder = buffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipelines.copyWeights)
        encoder.setArguments(
            .buffer(weights),
            .buffer(buffers.weightsFull))
        encoder.dispatchThreads(
            MTLSizeMake(config.parameterCount, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(pipelines.copyWeights.maxTotalThreadsPerThreadgroup, 1, 1))
        encoder.endEncoding()
    }
    
    private func resetStatistics(on buffer: MTLCommandBuffer) {
        iteration = 0
        if let encoder = buffer.makeBlitCommandEncoder() {
            encoder.fill(buffer: buffers.firstMoment, range: 0..<buffers.firstMoment.length, value: 0)
            encoder.fill(buffer: buffers.secondMoment, range: 0..<buffers.secondMoment.length, value: 0)
            encoder.endEncoding()
        }
    }
    
    /// gradients are assumed to be in Float32 precision
    public func step(on buffer: TimedCommandBuffer, gradients: MTLBuffer) {
        iteration += 1
        if let encoder = buffer.makeComputeCommandEncoder(named: "adam_step") {
            encoder.setComputePipelineState(pipelines.performStep)
            encoder.setArguments(
                .int(iteration),
                .float(config.learningRate),
                .float(config.weightDecay),
                .buffer(gradients),
                .buffer(buffers.weightsFull),
                .buffer(weights),
                .buffer(buffers.firstMoment),
                .buffer(buffers.secondMoment)
            )
            encoder.dispatchThreads(
                MTLSizeMake(config.parameterCount, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(pipelines.performStep.maxTotalThreadsPerThreadgroup, 1, 1))
            encoder.endEncoding()
        }
    }
}
