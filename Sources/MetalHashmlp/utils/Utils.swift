import Metal

func powi(_ base: UInt32, _ exponent: UInt32) -> UInt32 {
    var result: UInt32 = 1
    for _ in 0..<exponent {
        result *= base
    }
    return result
}

public enum ComputeArgument {
    case int(Int)
    case float(Float)
    case buffer(MTLBuffer, offset: Int = 0)
}

public class TimedCommandBuffer {
    private let commandBuffer: MTLCommandBuffer
    private let sampleBuffer: MTLCounterSampleBuffer
    private var commands: [String]
    
    public init(commandBuffer: MTLCommandBuffer, sampleBuffer: MTLCounterSampleBuffer) {
        self.commandBuffer = commandBuffer
        self.sampleBuffer = sampleBuffer
        self.commands = []
    }
    
    public func makeComputeCommandEncoder() -> MTLComputeCommandEncoder? {
        return commandBuffer.makeComputeCommandEncoder()
    }
    
    public func makeComputeCommandEncoder(named name: String) -> MTLComputeCommandEncoder? {
        let desc = MTLComputePassDescriptor()
        let attachment = desc.sampleBufferAttachments[0]!
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = 2 * commands.count + 0
        attachment.endOfEncoderSampleIndex = 2 * commands.count + 1
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder(descriptor: desc) else { return nil }
        encoder.label = name
        self.commands.append(name)
        return encoder
    }
    
    public func makeBlitCommandEncoder() -> MTLBlitCommandEncoder? {
        return commandBuffer.makeBlitCommandEncoder()
    }
    
    public func output() {
        let numCounters = 2 * commands.count
        let counters = try! sampleBuffer.resolveCounterRange(0..<numCounters)!
        counters.withUnsafeBytes { ptr in
            let data = ptr.assumingMemoryBound(to: MTLCounterResultTimestamp.self)
            let timestamps = (0..<numCounters).map { (i: Int) in Double(data[i].timestamp) / Double(NSEC_PER_SEC) }
            let total = Float(timestamps.max()! - timestamps.min()!)
            
            for (i, command) in commands.enumerated() {
                let time = Float(timestamps[2 * i + 1] - timestamps[2 * i + 0])
                print("  \(command): \(time * 1000) ms")
            }
            print("  total: \(total * 1000) ms")
            print()
        }
    }
}

public extension MTLDevice {
    func makeBuffer(named name: String, length: Int) -> MTLBuffer? {
        let buffer = makeBuffer(length: length)
        buffer?.label = name
        return buffer
    }
}

public extension MTLLibrary {
    func makePipeline(named name: String) throws -> MTLComputePipelineState {
        return try device.makeComputePipelineState(function: makeFunction(name: name)!)
    }
}

public extension MTLComputeCommandEncoder {
    func setArguments(_ args: ComputeArgument...) {
        for (argIndex, arg) in args.enumerated() {
            switch arg {
            case .int(let v):
                var p = UInt32(v)
                setBytes(&p, length: MemoryLayout<UInt32>.size, index: argIndex)
            case .float(let  v):
                var p = Float32(v)
                setBytes(&p, length: MemoryLayout<Float32>.size, index: argIndex)
            case .buffer(let buffer, let offset):
                setBuffer(buffer, offset: offset, index: argIndex)
            }
        }
    }
}

public enum FloatPrecision {
    case half
    case float
    
    public var size: Int {
        switch self {
        case .half: MemoryLayout<Float16>.size
        case .float: MemoryLayout<Float32>.size
        }
    }
}
