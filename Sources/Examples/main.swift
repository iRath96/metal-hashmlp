import Metal
import MetalKit
import MetalHashmlp
import simd
import UniformTypeIdentifiers

func saveImage(_ texture: MTLTexture) {
    let url = URL.currentDirectory().appending(component: "mlp.png")
    print("saving image as \(url)")
    
    let options = [CIImageOption.colorSpace: CGColorSpaceCreateDeviceRGB(),
               CIContextOption.outputPremultiplied: true,
               CIContextOption.useSoftwareRenderer: false] as! [CIImageOption : Any]
    guard let ciimage = CIImage(mtlTexture: texture, options: options) else {
        print("CIImage not created")
        return
    }
    let flipped = ciimage.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
    guard let cgImage = CIContext().createCGImage(flipped,
                                 from: flipped.extent,
                                 format: CIFormat.RGBA8,
                                 colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)  else {
        print("CGImage not created")
        return
    }
    if let imageDestination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
        CGImageDestinationAddImage(imageDestination, cgImage, nil)
        CGImageDestinationFinalize(imageDestination)
    }
}

let device = MTLCopyAllDevices().first!
let commandQueue = device.makeCommandQueue()!

let image = Bundle.module.url(forResource: "albert", withExtension: "jpg")!
let textureLoader = MTKTextureLoader(device: device)
let inputTexture = try! textureLoader.newTexture(URL: image, options: [
    .SRGB: false
])
//print(inputTexture)

let outputTexDesc = MTLTextureDescriptor()
outputTexDesc.width = inputTexture.width
outputTexDesc.height = inputTexture.height
outputTexDesc.pixelFormat = inputTexture.pixelFormat
outputTexDesc.usage = [.shaderRead, .shaderWrite]
let outputTexture = device.makeTexture(descriptor: outputTexDesc)!

// MARK: - configuration

let precision = FloatPrecision.half

let hashgrid = try! Hashgrid(on: device, with: .init(
    inputDim: 2,
    levels: 16,
    featuresPerLevel: 2,
    baseResolution: 16,
    perLevelScale: 1.5,
    hashmapSize: 1<<15,
    precision: precision))
let mlp = try! MultiLayerPerceptron(on: device, with: .init(
    inputNeurons: hashgrid.config.outputDim,
    hiddenLayers: 2,
    hiddenNeurons: 64,
    outputNeurons: 8,
    precision: precision))

let batchSize = 1<<18
let mlpTraining = mlp.batch(sized: batchSize)
let hgTraining = hashgrid.batch(
    sized: batchSize,
    output: mlpTraining.inputs,
    outputGradients: mlpTraining.inputGradients)

let mlpOptimizer = try! AdamOptimizer(for: mlp.weights, with: .init(
    parameterCount: mlp.config.weightCount,
    precision: precision,
    learningRate: 1e-2,
    weightDecay: 1e-6))
let hgOptimizer = try! AdamOptimizer(for: hashgrid.weights, with: .init(
    parameterCount: hashgrid.weightCount,
    precision: precision,
    learningRate: 1e-2,
    weightDecay: 1e-6))

let task = try! Task(on: device, inputTexture: inputTexture, outputTexture: outputTexture, config: .init(
    inputDim: hashgrid.config.inputDim,
    outputDim: mlp.config.outputNeurons,
    precision: precision))

let targets = device.makeBuffer(length: precision.size * mlpTraining.size * mlp.config.outputNeurons)!
if let buffer = commandQueue.makeCommandBuffer() {
    if let encoder = buffer.makeBlitCommandEncoder() {
        encoder.fill(buffer: targets, range: 0..<targets.length, value: 0)
        encoder.endEncoding()
    }
    buffer.commit()
    buffer.waitUntilCompleted()
}

let sbd = MTLCounterSampleBufferDescriptor()
sbd.counterSet = device.counterSets!.first!
sbd.sampleCount = 64
let sampleBuffer = try! device.makeCounterSampleBuffer(descriptor: sbd)

let capture = MTLCaptureDescriptor()
let manager = MTLCaptureManager.shared()
capture.captureObject = device

let debug = !true
let Nruns = debug ? 8 : 1000
for run in 1..<(Nruns+1) {
    let shouldCapture = debug && run == Nruns
    
    if shouldCapture {
        try! manager.startCapture(with: capture)
    }
    
    if let buffer = commandQueue.makeCommandBuffer() {
        let timedBuffer = TimedCommandBuffer(commandBuffer: buffer, sampleBuffer: sampleBuffer)
        
        task.randomInputs(on: timedBuffer, batchSize: mlpTraining.size, inputs: hgTraining.inputs, targets: targets)
        
        hgTraining.forward(on: timedBuffer)
        mlpTraining.forward(on: timedBuffer)
        
        task.computeLoss(on: timedBuffer,
            batchSize: mlpTraining.size,
            targets: targets,
            outputs: mlpTraining.outputs,
            outputGradients: mlpTraining.outputGradients)
        
        mlpTraining.backward(on: timedBuffer)
        hgTraining.backward(on: timedBuffer)
        
        mlpOptimizer.step(on: timedBuffer, gradients: mlp.weightGradients)
        hgOptimizer.step(on: timedBuffer, gradients: hashgrid.weightGradients)
        
        buffer.commit()
        buffer.waitUntilCompleted()
        
        if shouldCapture {
            manager.stopCapture()
        }
        
        print("RUN \(run)")
        timedBuffer.output()
        
        print("  loss: \(task.loss)")
        print()
        if task.loss.isNaN {
            mlpTraining.debug()
            exit(1)
        }
    }
}

if let buffer = commandQueue.makeCommandBuffer() {
    let inferenceBatchSize = outputTexture.width * outputTexture.height
    let mlpInference = mlp.batch(sized: inferenceBatchSize)
    let hgInference = hashgrid.batch(
        sized: inferenceBatchSize,
        output: mlpInference.inputs,
        outputGradients: mlpInference.inputGradients)
    
    let timedBuffer = TimedCommandBuffer(commandBuffer: buffer, sampleBuffer: sampleBuffer)
    
    task.inferenceInputs(on: timedBuffer, inputs: hgInference.inputs)
    hgInference.forward(on: timedBuffer)
    mlpInference.forward(on: timedBuffer)
    task.generateOutputs(on: timedBuffer, from: mlpInference.outputs)
    
    buffer.commit()
    buffer.waitUntilCompleted()
    
    timedBuffer.output()
    
    saveImage(outputTexture)
}
