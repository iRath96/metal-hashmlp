import Foundation
import Metal

public class MultiLayerPerceptron {
    public struct Configuration {
        public let inputNeurons: Int /// must be a multiple of 8
        public let hiddenLayers: Int /// must be at least one
        public let hiddenNeurons: Int /// must be a multiple of 8
        public let outputNeurons: Int /// must be a multiple of 8
        
        public let precision: FloatPrecision
        
        public var weightCount: Int {
            inputNeurons * hiddenNeurons +
            (hiddenLayers - 1) * hiddenNeurons * hiddenNeurons +
            hiddenNeurons * outputNeurons
        }
        
        public var activationsPerElement: Int { inputNeurons + hiddenLayers * hiddenNeurons + outputNeurons }
        
        public init(inputNeurons: Int, hiddenLayers: Int, hiddenNeurons: Int, outputNeurons: Int, precision: FloatPrecision) {
            self.inputNeurons = inputNeurons
            self.hiddenLayers = hiddenLayers
            self.hiddenNeurons = hiddenNeurons
            self.outputNeurons = outputNeurons
            self.precision = precision
        }
    }
    
    public class Batch {
        fileprivate let mlp: MultiLayerPerceptron
        
        public let size: Int /// must be a multiple of 8
        fileprivate let activations: MTLBuffer
        fileprivate let activationGradients: MTLBuffer
        
        public var config: Configuration { mlp.config }
        fileprivate var activationCount: Int { config.activationsPerElement * size }
        
        fileprivate init(mlp: MultiLayerPerceptron, size: Int, activations: MTLBuffer, activationGradients: MTLBuffer) {
            self.mlp = mlp
            self.size = size
            self.activations = activations
            self.activationGradients = activationGradients
        }
        
        fileprivate var outputOffset: Int {
            get {
                config.precision.size * size * (
                    config.inputNeurons +
                    config.hiddenLayers * config.hiddenNeurons
                )
            }
        }
        
        public var inputs: ComputeArgument {
            get { .buffer(activations, offset: 0) }
        }
        
        public var outputs: ComputeArgument {
            get { .buffer(activations, offset: outputOffset) }
        }
        
        public var inputGradients: ComputeArgument {
            get { .buffer(activationGradients, offset: 0) }
        }
        
        public var outputGradients: ComputeArgument {
            get { .buffer(activationGradients, offset: outputOffset) }
        }
        
        public func forward(on buffer: TimedCommandBuffer) { mlp.forward(on: buffer, for: self) }
        public func backward(on buffer: TimedCommandBuffer) { mlp.backward(on: buffer, for: self) }
        public func debug() { mlp.debug(for: self) }
    }
    
    private struct Pipelines {
        let randomizeWeights: MTLComputePipelineState
        let forwardPass: MTLComputePipelineState
        let backwardActivations: MTLComputePipelineState
        let backwardWeightsInput: MTLComputePipelineState
        let backwardWeightsHidden: MTLComputePipelineState
        let backwardWeightsOutput: MTLComputePipelineState
    }
    
    private struct Buffers {
        let weights: MTLBuffer
        let weightGradients: MTLBuffer /// in full precision
    }
    
    public let config: Configuration
    private let device: MTLDevice
    private let pipelines: Pipelines
    private let buffers: Buffers
    
    public var weights: MTLBuffer { buffers.weights }
    public var weightGradients: MTLBuffer { buffers.weightGradients }
    
    public init(on device: MTLDevice, with config: Configuration) throws {
        self.device = device
        self.config = config
        
        // MARK: setup pipelines
        
        let kernelConfig = """
constexpr constant uint InputDim = \(config.inputNeurons);
constexpr constant uint OutputDim = \(config.outputNeurons);
constexpr constant uint HiddenNeurons = \(config.hiddenNeurons);
constexpr constant uint HiddenLayers = \(config.hiddenLayers);
constexpr constant bool StoreActivations = true;

using FloatMLP = \(config.precision.size == 2 ? "half" : "float");
using FloatFP = float;

"""
        
        let kernelPath = Bundle.module.url(forResource: "mlp", withExtension: "metal", subdirectory: "metal")!
        let kernelSource = try! String(contentsOf: kernelPath)
        let library = try! device.makeLibrary(
            source: kernelConfig + kernelSource,
            options: MTLCompileOptions())
        
        self.pipelines = .init(
            randomizeWeights:      try library.makePipeline(named: "mlp_random_weights"),
            forwardPass:           try library.makePipeline(named: "mlp_forward"),
            backwardActivations:   try library.makePipeline(named: "mlp_backward_activations"),
            backwardWeightsInput:  try library.makePipeline(named: "mlp_backward_weights_input"),
            backwardWeightsHidden: try library.makePipeline(named: "mlp_backward_weights_hidden"),
            backwardWeightsOutput: try library.makePipeline(named: "mlp_backward_weights_output"))
        
        // MARK: setup buffers
        
        let weightsSize = config.precision.size * config.weightCount
        let weightGradientsSize = MemoryLayout<Float>.size * config.weightCount
        
        self.buffers = .init(
            weights:          device.makeBuffer(named: "MLP Weights",          length: weightsSize)!,
            weightGradients:  device.makeBuffer(named: "MLP Weight Gradients", length: weightGradientsSize)!)
        
        // MARK: initialize buffers
        
        let buffer = device.makeCommandQueue()!.makeCommandBuffer()!
        initializeWeights(on: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
    }
    
    private func initializeWeights(on buffer: MTLCommandBuffer) {
        // TODO: use individual scales for each weight matrix
        let encoder = buffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipelines.randomizeWeights)
        encoder.setArguments(
            .int(1337),
            .float(sqrtf(Float(6) / Float(config.hiddenNeurons * config.hiddenNeurons))),
            .buffer(buffers.weights)
        )
        encoder.dispatchThreads(
            MTLSizeMake(config.weightCount, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(
                pipelines.randomizeWeights.maxTotalThreadsPerThreadgroup, 1, 1))
        encoder.endEncoding()
    }
    
    public func batch(sized batchSize: Int) -> Batch {
        let activationsSize = config.precision.size * config.activationsPerElement * batchSize
        return .init(
            mlp: self,
            size: batchSize,
            activations: device.makeBuffer(named: "MLP Activations", length: activationsSize)!,
            activationGradients: device.makeBuffer(named: "MLP Activation Gradients", length: activationsSize)!)
    }
    
    fileprivate func forward(on buffer: TimedCommandBuffer, for batch: Batch) {
        if let encoder = buffer.makeComputeCommandEncoder(named: "mlp_forward") {
            encoder.setComputePipelineState(pipelines.forwardPass)
            
            encoder.setArguments(
                .int(batch.size),
                .buffer(buffers.weights),
                .buffer(batch.activations)
            )
            
            let threadgroupSize = 512 // TODO: pipeline.maxTotalThreadsPerThreadgroup
            encoder.dispatchThreads(
                MTLSizeMake(batch.size, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1))
            encoder.endEncoding()
        }
    }
    
    fileprivate func backward(on buffer: TimedCommandBuffer, for batch: Batch) {
        /// zero out gradients
        if let encoder = buffer.makeBlitCommandEncoder() {
            encoder.fill(buffer: buffers.weightGradients, range: 0..<buffers.weightGradients.length, value: 0)
            encoder.endEncoding()
        }
        
        if let encoder = buffer.makeComputeCommandEncoder(named: "mlp_grad_act") {
            encoder.setComputePipelineState(pipelines.backwardActivations)

            encoder.setArguments(
                .int(batch.size),
                .buffer(buffers.weights),
                .buffer(batch.activations),
                .buffer(batch.activationGradients)
            )
            
            let threadgroupSize = 512 // TODO: pipeline.maxTotalThreadsPerThreadgroup
            encoder.dispatchThreads(
                MTLSizeMake(batch.size, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1))
            encoder.endEncoding()
        }
        
        for layer in 0..<(config.hiddenLayers + 1) {
            if let encoder = buffer.makeComputeCommandEncoder(named: "mlp_grad_w\(layer)") {
                encoder.setComputePipelineState(
                    layer == 0 ? pipelines.backwardWeightsInput :
                    layer == config.hiddenLayers ? pipelines.backwardWeightsOutput :
                    pipelines.backwardWeightsHidden
                )
                
                encoder.setArguments(
                    .int(batch.size),
                    .int(layer),
                    .buffer(batch.activations),
                    .buffer(batch.activationGradients),
                    .buffer(buffers.weightGradients)
                )
                
                let threadgroupSize = 32 // TODO: ???
                let gridSize = 256 * threadgroupSize // TODO: ???
                encoder.dispatchThreads(
                    MTLSizeMake(gridSize, 1, 1),
                    threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1))
                encoder.endEncoding()
            }
        }
    }
    
    fileprivate func debug(for batch: Batch) {
        switch config.precision {
#if arch(arm64)
        case .half:  debug(floatType: Float16.self, for: batch)
#else
        case .half:  print("cannot debug float16 on non-arm64 architecture")
#endif
        case .float: debug(floatType: Float32.self, for: batch)
        }
    }
    
    private func debug<T>(floatType: T.Type, for batch: Batch) {
        func layer_dim(_ layer: Int) -> Int {
            return layer == 0 ? config.inputNeurons :
                layer == config.hiddenLayers + 1 ? config.outputNeurons :
                config.hiddenNeurons
        }
        
        for layer in 0..<(config.hiddenLayers + 1) {
            print()
            print("layer \(layer) -> \(layer + 1)")
            print()
            
            let prev_dim = layer_dim(layer)
            let next_dim = layer_dim(layer + 1)
            
            let offset = layer == 0 ? 0 :
                config.inputNeurons * config.hiddenNeurons +
                (layer - 1) * config.hiddenNeurons * config.hiddenNeurons
            
            let wptr = buffers.weights.contents()
                .assumingMemoryBound(to: floatType)
                .advanced(by: offset)
            print("a")
            for prev in 0..<prev_dim {
                for next in 0..<next_dim {
                    print(wptr[next * prev_dim + prev], terminator: "\t")
                }
                print()
            }
            print()
            
            let gptr = buffers.weightGradients.contents()
                .assumingMemoryBound(to: Float.self)
                .advanced(by: offset)
            print("g")
            for prev in 0..<prev_dim {
                for next in 0..<next_dim {
                    print(gptr[next * prev_dim + prev], terminator: "\t")
                }
                print()
            }
            print()
        }
        
        for layer in 0..<(config.hiddenLayers + 2) {
            let offset = layer == 0 ? 0 : config.inputNeurons + (layer - 1) * config.hiddenNeurons
            let current_dim = layer_dim(layer)
            let aptr = batch.activations.contents()
                .assumingMemoryBound(to: floatType)
                .advanced(by: batch.size * offset)
            let dptr = batch.activationGradients.contents()
                .assumingMemoryBound(to: floatType)
                .advanced(by: batch.size * offset)
            
            print()
            print("layer \(layer)")
            print()
            
            //for i in 0..<batch.size {
            for i in 0..<min(batch.size, 4) {
            //for i in (batch_size-4)..<batch_size {
                print("a[\(i)]:", terminator: "\t")
                for dim in 0..<current_dim {
                    print(aptr[i * current_dim + dim], terminator: "\t")
                }
                print()
                print("g[\(i)]:", terminator: "\t")
                for dim in 0..<current_dim {
                    print(dptr[i * current_dim + dim], terminator: "\t")
                }
                print()
            }
        }
    }
}
