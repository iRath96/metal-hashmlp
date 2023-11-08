import Foundation
import Metal

public class Hashgrid {
    public struct Configuration {
        public let inputDim: Int
        public let levels: Int
        public let featuresPerLevel: Int
        public let baseResolution: Int
        public let perLevelScale: Float
        public let hashmapSize: Int
        public let precision: FloatPrecision
        
        public var outputDim: Int { levels * featuresPerLevel }
        public var log2PerLevelScale: Float { log2(perLevelScale) }
        
        public init(inputDim: Int, levels: Int, featuresPerLevel: Int, baseResolution: Int, perLevelScale: Float, hashmapSize: Int, precision: FloatPrecision) {
            self.inputDim = inputDim
            self.levels = levels
            self.featuresPerLevel = featuresPerLevel
            self.baseResolution = baseResolution
            self.perLevelScale = perLevelScale
            self.hashmapSize = hashmapSize
            self.precision = precision
        }
    }
    
    public class Batch {
        fileprivate let hashgrid: Hashgrid
        
        public let size: Int /// must be a multiple of 8
        public let inputs: MTLBuffer
        public let outputs: ComputeArgument
        public let outputGradients: ComputeArgument
        
        public var config: Configuration { hashgrid.config }
        
        public init(hashgrid: Hashgrid, size: Int, inputs: MTLBuffer, outputs: ComputeArgument, outputGradients: ComputeArgument) {
            self.hashgrid = hashgrid
            self.size = size
            self.inputs = inputs
            self.outputs = outputs
            self.outputGradients = outputGradients
        }
        
        public func forward(on buffer: TimedCommandBuffer) { hashgrid.forward(on: buffer, for: self) }
        public func backward(on buffer: TimedCommandBuffer) { hashgrid.backward(on: buffer, for: self) }
    }
    
    private struct Pipelines {
        let randomizeWeights: MTLComputePipelineState
        let forward: MTLComputePipelineState
        let backward: MTLComputePipelineState
    }
    
    private struct Buffers {
        let offsetTable: MTLBuffer
        let weights: MTLBuffer
        let weightGradients: MTLBuffer /// always in Float32 precision
    }
    
    public let config: Configuration
    private let device: MTLDevice
    private let pipelines: Pipelines
    private let buffers: Buffers
    public let weightCount: Int
    
    public var weights: MTLBuffer { buffers.weights }
    public var weightGradients: MTLBuffer { buffers.weightGradients }
    
    public init(on device: MTLDevice, with config: Configuration) throws {
        self.device = device
        self.config = config
        
        let kernelConfig = """
constexpr constant uint InputDim = \(config.inputDim);
constexpr constant uint FeaturesPerLevel = \(config.featuresPerLevel);
constexpr constant uint FeaturesPerThread = \(config.featuresPerLevel);
constexpr constant uint Levels = \(config.levels);

using FloatMLP = \(config.precision.size == 2 ? "half" : "float");
using FloatFP = float;

"""
        
        let kernelPath = Bundle.module.url(forResource: "hashgrid", withExtension: "metal", subdirectory: "metal")!
        let kernelSource = try! String(contentsOf: kernelPath)
        let library = try! device.makeLibrary(
            source: kernelConfig + kernelSource,
            options: MTLCompileOptions())
        
        self.pipelines = .init(
            randomizeWeights: try library.makePipeline(named: "random_weights"),
            forward:          try library.makePipeline(named: "forward"),
            backward:         try library.makePipeline(named: "backward"))
        
        let offsetTableBuffer = device.makeBuffer(
            named: "Hashgrid Offset Table",
            length: MemoryLayout<UInt32>.size * (config.levels + 1))!
        
        self.weightCount = Hashgrid.buildOffsetTable(into: offsetTableBuffer, for: config) * config.featuresPerLevel
        self.buffers = .init(
            offsetTable: offsetTableBuffer,
            weights: device.makeBuffer(
                named: "Hashgrid Weights",
                length: config.precision.size * self.weightCount)!,
            weightGradients: device.makeBuffer(
                named: "Hashgrid Weight Gradients",
                length: MemoryLayout<Float>.size * self.weightCount)!)
        
        let buffer = device.makeCommandQueue()!.makeCommandBuffer()!
        initializeWeights(on: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
    }
    
    private static func buildOffsetTable(into buffer: MTLBuffer, for config: Configuration) -> Int {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        let maxParams = UInt32.max / 2
        var offset: UInt32 = 0
        for level in 0..<config.levels {
            let scale = exp2(Float(level) * config.log2PerLevelScale) * Float(config.baseResolution) - Float(1)
            let resolution = UInt32(ceil(scale) + 1)
            var paramsInLevel = pow(Float(resolution), Float(config.inputDim)) > Float(maxParams) ?
                maxParams :
                powi(resolution, UInt32(config.inputDim))
            paramsInLevel = min(paramsInLevel, UInt32(config.hashmapSize))
            
            ptr[level] = offset
            print("offset_table[\(level)] = \(offset)")
            offset += paramsInLevel
        }
        ptr[config.levels] = offset
        print("final offset: \(offset)")
        return Int(offset)
    }
    
    private func initializeWeights(on buffer: MTLCommandBuffer) {
        let encoder = buffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipelines.randomizeWeights)
        encoder.setArguments(
            .int(1337),
            .float(1e-4),
            .buffer(buffers.weights)
        )
        encoder.dispatchThreads(
            MTLSizeMake(weightCount, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(pipelines.randomizeWeights.maxTotalThreadsPerThreadgroup, 1, 1))
        encoder.endEncoding()
    }
    
    public func batch(sized batchSize: Int, output: ComputeArgument, outputGradients: ComputeArgument) -> Batch {
        let inputSize = config.precision.size * config.inputDim * batchSize
        return .init(
            hashgrid: self,
            size: batchSize,
            inputs: device.makeBuffer(named: "Hashgrid Inputs", length: inputSize)!,
            outputs: output,
            outputGradients: outputGradients)
    }
    
    fileprivate func forward(on buffer: TimedCommandBuffer, for batch: Batch) {
       if let encoder = buffer.makeComputeCommandEncoder(named: "hg_forward") {
            encoder.setComputePipelineState(pipelines.forward)
            
            encoder.setArguments(
                .int(batch.size),
                .buffer(buffers.offsetTable),
                .int(config.baseResolution),
                .float(config.log2PerLevelScale),
                
                .buffer(buffers.weights),
                .buffer(batch.inputs),
                batch.outputs
            )
            
            encoder.dispatchThreads(
                MTLSizeMake(batch.size, config.levels, 1),
                threadsPerThreadgroup: MTLSizeMake(pipelines.forward.maxTotalThreadsPerThreadgroup, 1, 1))
            encoder.endEncoding()
        }
    }
    
    fileprivate func backward(on buffer: TimedCommandBuffer, for batch: Batch) {
        if let encoder = buffer.makeBlitCommandEncoder() {
            encoder.fill(buffer: buffers.weightGradients, range: 0..<buffers.weightGradients.length, value: 0)
            encoder.endEncoding()
        }
        
        if let encoder = buffer.makeComputeCommandEncoder(named: "hg_backward") {
            encoder.setComputePipelineState(pipelines.backward)
            
            encoder.setArguments(
                .int(batch.size),
                .buffer(buffers.offsetTable),
                .int(config.baseResolution),
                .float(config.log2PerLevelScale),
                
                .buffer(buffers.weightGradients),
                .buffer(batch.inputs),
                batch.outputGradients
            )
            
            encoder.dispatchThreads(
                MTLSizeMake(batch.size, config.levels, 1),
                threadsPerThreadgroup: MTLSizeMake(pipelines.backward.maxTotalThreadsPerThreadgroup, 1, 1))
            encoder.endEncoding()
        }
    }
}
