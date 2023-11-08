#include <metal_stdlib>
#define unrolled _Pragma("clang loop unroll(full)")

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

inline float grid_scale(uint32_t level, float log2_per_level_scale, uint32_t base_resolution) {
    // The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
    // than the number of cells. This is slightly different from the notation in the paper,
    // but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
    return exp2(level * log2_per_level_scale) * base_resolution - 1;
}

inline uint32_t grid_resolution(float scale) {
    return (uint32_t)ceil(scale) + 1;
}

inline void pos_fract(
    const float input, thread float& pos, thread uint32_t& pos_grid, float scale
) {
    // The offset of 0.5 causes different scales to be staggered with respect to each other, thus
    // preventing spurious alignment of fractional coordinates upon integer scales (or powers thereof).
    // This is mentioned in Appendix A of the "Instant Neural Graphics Primitives" paper.
    // The offset can cause wraparound indexing in dense grids, which didn't negatively impact
    // the approximation quality in any of our tests.
    pos = fma(scale, input, 0.5f);
    float tmp = floor(pos);
    pos_grid = (uint32_t)(int)tmp;
    pos -= tmp;
}

template <uint32_t N_DIMS, uint32_t N_PRIMES>
uint32_t lcg_hash(const uint32_t pos_grid[N_DIMS], const uint32_t primes[N_PRIMES]) {
    static_assert(N_DIMS <= N_PRIMES, "lcg_hash can only hash up to N_PRIMES dimensions.");

    uint32_t result = 0;

    for (uint32_t i = 0; i < N_DIMS; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}

uint32_t coherent_prime_hash(const uint32_t pos_grid[InputDim]) {
    constexpr uint32_t factors[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
    return lcg_hash<InputDim, 7>(pos_grid, factors);
}

uint32_t grid_index(
    const uint32_t hashmap_size, const uint32_t grid_resolution, const uint32_t pos_grid[InputDim]
) {
    uint32_t stride = 1;
    uint32_t index = 0;

    // The second part of the loop condition is needed to avoid integer overflows in finer levels.
    for (uint32_t dim = 0; dim < InputDim && stride <= hashmap_size; ++dim) {
        index += pos_grid[dim] * stride;
        stride *= grid_resolution;
    }

    if (hashmap_size < stride) {
        index = coherent_prime_hash(pos_grid);
    }

    return index % hashmap_size;
}

kernel void forward(
    constant uint&  num_elements,
    constant uint*  offset_table,
    constant uint&  base_resolution,
    constant float& log2_per_level_scale,
    
    constant FloatMLP* grid,
    constant FloatMLP* positions_in,
    device FloatMLP* encoded_positions,
    
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 blockDim [[threads_per_threadgroup]]
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    
    const uint32_t level = blockIdx.y;
    
    struct {
        constant FloatMLP* grid;
        uint32_t hashmap_size;
        float scale;
        uint32_t resolution;
        
        constant FloatMLP* operator()(uint32_t local_pos[InputDim]) {
            const uint32_t index = grid_index(hashmap_size, resolution, local_pos) * FeaturesPerLevel;
            return grid + index;
        }
    } grid_val;
    
    grid += offset_table[level] * FeaturesPerLevel;
    grid_val.grid = grid;
    grid_val.hashmap_size = offset_table[level + 1] - offset_table[level];
    
    grid_val.scale = grid_scale(level, log2_per_level_scale, base_resolution);
    grid_val.resolution = grid_resolution(grid_val.scale);
 
    float pos[InputDim];
    uint32_t pos_grid[InputDim];
    
    unrolled for (uint32_t dim = 0; dim < InputDim; ++dim) {
        pos_fract(positions_in[i + dim * num_elements], pos[dim], pos_grid[dim], grid_val.scale);
    }
    
    FloatMLP result[FeaturesPerLevel];
    unrolled for (uint f = 0; f < FeaturesPerLevel; f++) result[f] = 0;
    
    unrolled for (uint32_t idx = 0; idx < (1 << InputDim); ++idx) {
        FloatMLP weight = 1;
        uint32_t pos_grid_local[InputDim];
        
        unrolled for (uint32_t dim = 0; dim < InputDim; ++dim) {
            if ((idx & (1 << dim)) == 0) {
                weight *= 1 - pos[dim];
                pos_grid_local[dim] = pos_grid[dim];
            } else {
                weight *= pos[dim];
                pos_grid_local[dim] = pos_grid[dim] + 1;
            }
        }
        
        constant FloatMLP* g = grid_val(pos_grid_local);
        unrolled for (uint f = 0; f < FeaturesPerLevel; f++) {
            result[f] = fma(weight, g[f], result[f]);
        }
    }
    
    unrolled for (uint32_t f = 0; f < FeaturesPerLevel; ++f) {
        //encoded_positions[i + (level * FeaturesPerLevel + f) * num_elements] = result[f];
        encoded_positions[i * Levels * FeaturesPerLevel + level * FeaturesPerLevel + f] = result[f];
    }
}

kernel void backward(
    constant uint&  num_elements,
    constant uint*  offset_table,
    constant uint&  base_resolution,
    constant float& log2_per_level_scale,
    
    device atomic_float* __restrict__ grid_gradient,
    const device FloatMLP* __restrict__ positions_in,
    device FloatMLP* __restrict__  dL_dy,
    
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 blockDim [[threads_per_threadgroup]]
) {
    const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * FeaturesPerThread) / FeaturesPerLevel;
    if (i >= num_elements) return;
    
    const uint32_t level = blockIdx.y; // <- the level is the same for all threads.
    const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * FeaturesPerThread - i * FeaturesPerLevel;
    
    grid_gradient += offset_table[level] * FeaturesPerLevel;
    const uint32_t hashmap_size = offset_table[level + 1] - offset_table[level];
    
    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);
    
    float pos[InputDim];
    uint32_t pos_grid[InputDim];
    
    unrolled for (uint dim = 0; dim < InputDim; dim++) {
        pos_fract(positions_in[i + dim * num_elements], pos[dim], pos_grid[dim], scale);
    }
    
    FloatMLP grad[FeaturesPerThread];
    unrolled for (uint f = 0; f < FeaturesPerThread; f++) {
        //grad[f] = dL_dy[i + (level * FeaturesPerLevel + feature + f) * num_elements];
        grad[f] = dL_dy[i * Levels * FeaturesPerLevel + level * FeaturesPerLevel + feature + f];
    }
    
    // N-linear interpolation
    unrolled for (uint idx = 0; idx < (1 << InputDim); idx++) {
        FloatMLP weight = 1;
        uint32_t pos_grid_local[InputDim];
        
        unrolled for (uint dim = 0; dim < InputDim; dim++) {
            if ((idx & (1 << dim)) == 0) {
                weight *= 1 - pos[dim];
                pos_grid_local[dim] = pos_grid[dim];
            } else {
                weight *= pos[dim];
                pos_grid_local[dim] = pos_grid[dim] + 1;
            }
        }
        
        // add_grid_gradient(pos_grid_local, grad, weight);
        uint index = grid_index(hashmap_size, resolution, pos_grid_local) * FeaturesPerLevel + feature;
        unrolled for (uint f = 0; f < FeaturesPerThread; f++) {
            atomic_fetch_add_explicit(grid_gradient + index + f, weight * grad[f], memory_order_relaxed);
            //((device float *)grid_gradient)[index + f] += weight * grad[f];
        }
    }
}

kernel void random_weights(
    constant uint& seed,
    constant float& scale,

    device FloatMLP* weights,
    
    uint i [[thread_position_in_grid]]
) {
    float weight = scale * (2 * sample_tea_float32(seed, i) - 1);
    weights[i] = FloatMLP(weight);
}
