#include <metal_stdlib>

inline uint32_t sample_tea_32(uint32_t v0, uint32_t v1, int rounds = 6) {
    uint32_t sum = 0;

    for (int i = 0; i < rounds; ++i) {
        sum += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v1;
}

inline float sample_tea_float32(uint32_t v0, uint32_t v1, int rounds = 6) {
    union {
        uint32_t raw;
        float f;
    } v;
    v.raw = (sample_tea_32(v0, v1, rounds) >> 9) | 0x3f800000u;
    return v.f - 1.f;
}

using namespace metal;

kernel void generate_inputs(
    constant int& seed,

    device FloatMLP *inputs,
    device FloatMLP *targets,
    
    texture2d<float> image,
    
    uint i [[thread_position_in_grid]],
    uint batch_size [[threads_per_grid]]
) {
    float2 pos;
    pos.x = sample_tea_float32(seed, 2 * i + 0);
    pos.y = sample_tea_float32(seed, 2 * i + 1);
    
    constexpr sampler s(coord::normalized);
    
    float4 color;
    color = image.sample(s, pos);

    inputs[i + 0 * batch_size] = FloatMLP(pos.x);
    inputs[i + 1 * batch_size] = FloatMLP(pos.y);

    targets[OutputDim * i + 0] = FloatMLP(color.x);
    targets[OutputDim * i + 1] = FloatMLP(color.y);
    targets[OutputDim * i + 2] = FloatMLP(color.z);
}

kernel void inference_inputs(
    device FloatMLP *inputs,
    
    uint2 coord [[thread_position_in_grid]],
    uint2 dim [[threads_per_grid]]
) {
    uint batch_size = dim.x * dim.y;
    uint i = coord.x + coord.y * dim.x;
    inputs[i + 0 * batch_size] = FloatMLP((coord.x + 0.5) / dim.x);
    inputs[i + 1 * batch_size] = FloatMLP((coord.y + 0.5) / dim.y);
}

kernel void generate_outputs(
    const device FloatMLP *outputs,
    
    texture2d<float, access::write> image,
    
    uint2 coord [[thread_position_in_grid]],
    uint2 dim [[threads_per_grid]]
) {
    uint i = coord.x + coord.y * dim.x;
    image.write(float4(
        outputs[OutputDim * i + 0],
        outputs[OutputDim * i + 1],
        outputs[OutputDim * i + 2],
        1
    ), coord);
}

kernel void mlp_loss(
    constant uint& batch_size,
    
    const device FloatMLP* targets,
    const device FloatMLP* activations,
    device FloatMLP* dL_dactivations,
    device atomic<FloatFP>* loss,
    
    uint threadIdx [[thread_position_in_grid]],
    uint gridDim [[threads_per_grid]]
) {
    const FloatFP loss_scale = 1 / FloatFP(OutputDim * batch_size);
    const FloatFP grad_scale = 1e+4;
    FloatFP loss_sum = 0;
    
    for (uint index = threadIdx; index < batch_size; index += gridDim) {
        for (uint dim = 0; dim < OutputDim; dim++) {
            FloatFP error = activations[OutputDim * index + dim] - targets[OutputDim * index + dim];
            FloatFP mse = loss_scale * error * error;
            FloatFP grad = loss_scale * 2 * error;
            
            dL_dactivations[OutputDim * index + dim] = FloatMLP(grad_scale * grad);
            loss_sum += mse;
        }
    }
    
    atomic_fetch_add_explicit(loss, loss_sum, memory_order_relaxed);
}
