import Metal
import MetalHashmlp

class Task {
    struct Configuration {
        public let inputDim: Int
        public let outputDim: Int
        public let precision: FloatPrecision
    }
    
    private struct Pipelines {
        let randomInputs: MTLComputePipelineState
        let regularInputs: MTLComputePipelineState
        let computeLoss: MTLComputePipelineState
        let generateOutputs: MTLComputePipelineState
    }
    
    private struct Buffers {
        let loss: MTLBuffer
    }
    
    let device: MTLDevice
    let inputTexture: MTLTexture
    let outputTexture: MTLTexture
    private let pipelines: Pipelines
    private let buffers: Buffers
    private var randomSeed = 0
    
    var loss: Float { buffers.loss.contents().assumingMemoryBound(to: Float.self)[0] }
    
    init(on device: MTLDevice, inputTexture: MTLTexture, outputTexture: MTLTexture, config: Configuration) throws {
        let kernelConfig = """
constexpr constant uint InputDim = \(config.inputDim);
constexpr constant uint OutputDim = \(config.outputDim);
using FloatMLP = \(config.precision.size == 2 ? "half" : "float");
using FloatFP = float;

"""
        
        let kernelPath = Bundle.module.url(forResource: "kernels", withExtension: "metal", subdirectory: "metal")!
        let kernelSource = try! String(contentsOf: kernelPath)
        let library = try! device.makeLibrary(
            source: kernelConfig + kernelSource,
            options: MTLCompileOptions())
        
        self.device = device
        self.inputTexture = inputTexture
        self.outputTexture = outputTexture
        self.pipelines = .init(
            randomInputs:    try library.makePipeline(named: "generate_inputs"),
            regularInputs:   try library.makePipeline(named: "inference_inputs"),
            computeLoss:     try library.makePipeline(named: "mlp_loss"),
            generateOutputs: try library.makePipeline(named: "generate_outputs"))
        self.buffers = .init(
            loss: device.makeBuffer(named: "Loss", length: MemoryLayout<Float>.size)!)
    }
    
    func randomInputs(on buffer: TimedCommandBuffer, batchSize: Int, inputs: MTLBuffer, targets: MTLBuffer) {
        randomSeed += 1
        if let encoder = buffer.makeComputeCommandEncoder(named: "generate_inputs") {
            encoder.setComputePipelineState(pipelines.randomInputs)
            encoder.setTexture(inputTexture, index: 0)
            
            encoder.setArguments(
                .int(randomSeed),
                .buffer(inputs),
                .buffer(targets)
            )
            
            encoder.dispatchThreads(
                MTLSizeMake(batchSize, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(pipelines.randomInputs.maxTotalThreadsPerThreadgroup, 1, 1))
            encoder.endEncoding()
        }
    }
    
    func inferenceInputs(on buffer: TimedCommandBuffer, inputs: MTLBuffer) {
        if let encoder = buffer.makeComputeCommandEncoder(named: "inference_inputs") {
            encoder.setComputePipelineState(pipelines.regularInputs)
            encoder.setArguments(
                .buffer(inputs)
            )
            encoder.dispatchThreads(
                MTLSizeMake(outputTexture.width, outputTexture.height, 1),
                threadsPerThreadgroup: MTLSizeMake(32, 32, 1))
            encoder.endEncoding()
        }
    }
    
    func generateOutputs(on buffer: TimedCommandBuffer, from outputs: ComputeArgument) {
        if let encoder = buffer.makeComputeCommandEncoder(named: "generate_outputs") {
            encoder.setComputePipelineState(pipelines.generateOutputs)
            encoder.setTexture(outputTexture, index: 0)
            encoder.setArguments(outputs)
            encoder.dispatchThreads(
                MTLSizeMake(outputTexture.width, outputTexture.height, 1),
                threadsPerThreadgroup: MTLSizeMake(32, 32, 1))
            encoder.endEncoding()
        }
    }
    
    func computeLoss(on buffer: TimedCommandBuffer, batchSize: Int, targets: MTLBuffer, outputs: ComputeArgument, outputGradients: ComputeArgument) {
        if let encoder = buffer.makeBlitCommandEncoder() {
            encoder.fill(buffer: buffers.loss, range: 0..<buffers.loss.length, value: 0)
            encoder.endEncoding()
        }
        
        if let encoder = buffer.makeComputeCommandEncoder(named: "mlp_loss") {
            encoder.setComputePipelineState(pipelines.computeLoss)
            
            encoder.setArguments(
                .int(batchSize),
                .buffer(targets),
                outputs,
                outputGradients,
                .buffer(buffers.loss)
            )
            
            let threadgroupSize = 32 // TODO: pipeline.maxTotalThreadsPerThreadgroup
            let gridSize = 256 * threadgroupSize
            encoder.dispatchThreads(
                MTLSizeMake(gridSize, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1))
            encoder.endEncoding()
        }
    }
}
