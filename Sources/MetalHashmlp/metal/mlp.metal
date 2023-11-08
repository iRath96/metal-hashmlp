#include <metal_stdlib>
#include <metal/metal_simdgroup_matrix>
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

using simdgroup_mlp8x8 = simdgroup_matrix<FloatMLP, 8, 8>;

#define SHARED_MEMORY_WEIGHTS 1

#ifdef SHARED_MEMORY_WEIGHTS
constexpr constant uint WeightsCount = (
    InputDim * HiddenNeurons +
    (HiddenLayers - 1) * HiddenNeurons * HiddenNeurons +
    HiddenNeurons * OutputDim
);
#endif

static_assert((InputDim % 8) == 0, "input dimensionality must be divisible by 8");
static_assert((OutputDim % 8) == 0, "output dimensionality must be divisible by 8");
static_assert((HiddenNeurons % 8) == 0, "hidden neurons must be divisible by 8");
static_assert(HiddenLayers >= 1, "at least one hidden layer must be present");

bfloat max(bfloat a, bfloat b) { return max(a, b); }

// ReLU
//inline FloatMLP activation_function(FloatMLP x) { return max(x, FloatMLP(0)); }
//inline FloatMLP activation_gradient(FloatMLP x) { return x > 0 ? 1 : 0; }

// Leaky ReLU
inline FloatMLP activation_function(FloatMLP x) { return max(x, FloatMLP(0.1) * x); }
inline FloatMLP activation_gradient(FloatMLP x) { return x < FloatMLP(0) ? FloatMLP(0.1) : FloatMLP(1); }

// Sigmoid
//inline FloatMLP activation_function(FloatMLP x) { return 1 / (1 + exp(-x)); }
//inline FloatMLP activation_gradient(FloatMLP x) { return x * (1 - x); }

// Softplus
//inline FloatMLP activation_function(FloatMLP x) { return log(1 + exp(x)); }
//inline FloatMLP activation_gradient(FloatMLP x) { return (exp(x) - 1) / exp(x); }

// activations are stored in `batch x dim' layout, row-major order
// -> transposed on load/store, i.e., working in `dim x batch' layout
// weight matrices are stored in next_dim x prev_dim layout, row-major order

template<uint PrevDimBy8, uint NextDimBy8, typename T>
void forward(
    const thread simdgroup_mlp8x8 (&prev)[PrevDimBy8], // prev_dim x batch
    thread simdgroup_mlp8x8 (&next)[NextDimBy8], // next_dim x batch
    thread T &cur_weights
) {
    unrolled for (uint out = 0; out < NextDimBy8; out++) {
        unrolled for (uint in = 0; in < PrevDimBy8; in++) {
            simdgroup_mlp8x8 weightFragment; // next_dim x prev_dim
            simdgroup_load(weightFragment, cur_weights + 8 * in + 8 * out * (8 * PrevDimBy8), 8 * PrevDimBy8);
            
            if (in == 0) simdgroup_multiply(next[out], weightFragment, prev[in]);
            else simdgroup_multiply_accumulate(next[out], weightFragment, prev[in], next[out]);
        }
    }
    cur_weights += (8 * PrevDimBy8) * (8 * NextDimBy8);
}

template<uint PrevDimBy8, uint NextDimBy8, typename T>
void backward(
    thread simdgroup_mlp8x8 (&prev)[PrevDimBy8], // prev_dim x batch
    const thread simdgroup_mlp8x8 (&next)[NextDimBy8], // next_dim x batch
    thread T &cur_weights
) {
    cur_weights -= (8 * NextDimBy8) * (8 * PrevDimBy8);
    unrolled for (uint out = 0; out < PrevDimBy8; out++) {
        unrolled for (uint in = 0; in < NextDimBy8; in++) {
            simdgroup_mlp8x8 weightFragment; // prev_dim x next_dim
            simdgroup_load(weightFragment, cur_weights + 8 * out + 8 * in * (8 * PrevDimBy8), 8 * PrevDimBy8, ulong2(0, 0), true);

            if (in == 0) simdgroup_multiply(prev[out], weightFragment, next[in]);
            else simdgroup_multiply_accumulate(prev[out], weightFragment, next[in], prev[out]);
        }
    }
}

template<int DimBy8>
void load(thread simdgroup_mlp8x8 (&layer)[DimBy8], const device FloatMLP* activations) {
    unrolled for (uint in = 0; in < DimBy8; in++) {
        simdgroup_load(layer[in], activations, 8 * DimBy8, ulong2(0, 0), true);
        activations += 8;
    }
}

template<int DimBy8>
void store(thread simdgroup_mlp8x8 (&layer)[DimBy8], device FloatMLP* activations) {
    unrolled for (uint out = 0; out < DimBy8; out++) {
        simdgroup_store(layer[out], activations, 8 * DimBy8, ulong2(0, 0), true);
        activations += 8;
    }
}

template<int DimBy8>
void activation_function(thread simdgroup_mlp8x8 (&layer)[DimBy8]) {
    unrolled for (uint in = 0; in < DimBy8; in++) {
        auto raw = (thread FloatMLP *)&layer[in];
        raw[0] = activation_function(raw[0]);
        raw[1] = activation_function(raw[1]);
    }
}

template<int DimBy8>
void activation_gradient(thread simdgroup_mlp8x8 (&layer)[DimBy8], const device FloatMLP* activations) {
    unrolled for (uint out = 0; out < DimBy8; out++) {
        simdgroup_mlp8x8 activationFragment;
        simdgroup_load(activationFragment, activations + 8 * out, 8 * DimBy8, ulong2(0, 0), true);
        
        auto localActivations = (thread FloatMLP *)&activationFragment;
        auto localGradient = (thread FloatMLP *)&layer[out];
        localGradient[0] *= activation_gradient(localActivations[0]);
        localGradient[1] *= activation_gradient(localActivations[1]);
    }
}

kernel void mlp_forward(
    constant uint& batch_size,
    
    const device FloatMLP* weights,
    device FloatMLP* activations,
    
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 blockDim [[threads_per_threadgroup]],
    uint simdIdx [[simdgroup_index_in_threadgroup]],
    uint simdDim [[threads_per_simdgroup]]
) {
#ifdef SHARED_MEMORY_WEIGHTS
    threadgroup FloatMLP smem_weights[WeightsCount];
    for (uint i = threadIdx.x; i < WeightsCount; i += blockDim.x) {
        smem_weights[i] = weights[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    
    uint index = blockDim.x * blockIdx.x + simdDim * simdIdx;
    
    for (uint subgroup = 0; subgroup < 4; subgroup++) {
        simdgroup_mlp8x8 input[InputDim/8];
        simdgroup_mlp8x8 hidden[HiddenLayers][HiddenNeurons/8];
        simdgroup_mlp8x8 output[OutputDim/8];

#ifdef SHARED_MEMORY_WEIGHTS
        const threadgroup FloatMLP* cur_weights = smem_weights;
#else
        const device FloatMLP* cur_weights = weights;
#endif

        load(input, activations + InputDim * index);

        forward(input, hidden[0], cur_weights);
        activation_function(hidden[0]);
        if (StoreActivations) {
            store(hidden[0], activations + HiddenNeurons * index + batch_size * InputDim);
        }
        unrolled for (uint layer = 1; layer < HiddenLayers; layer++) {
            forward(hidden[layer - 1], hidden[layer], cur_weights);
            activation_function(hidden[layer]);
            if (StoreActivations) {
                store(hidden[layer], activations + HiddenNeurons * index + batch_size * (InputDim + layer * HiddenNeurons));
            }
        }
        forward(hidden[HiddenLayers - 1], output, cur_weights);
        
        store(output, activations + OutputDim * index + batch_size * (
            InputDim + HiddenLayers * HiddenNeurons
        ));
        
        index += 8;
    }
}

kernel void mlp_backward_activations(
    constant uint& batch_size,
    
    const device FloatMLP* weights,
    const device FloatMLP* activations,
    device FloatMLP* dL_dactivations,
    
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 blockDim [[threads_per_threadgroup]],
    uint simdIdx [[simdgroup_index_in_threadgroup]],
    uint simdDim [[threads_per_simdgroup]]
) {
#ifdef SHARED_MEMORY_WEIGHTS
    threadgroup FloatMLP smem_weights[WeightsCount];
    for (uint i = threadIdx.x; i < WeightsCount; i += blockDim.x) {
        smem_weights[i] = weights[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    
    uint index = blockDim.x * blockIdx.x + simdDim * simdIdx;
    
    for (uint subgroup = 0; subgroup < 4; subgroup++) {
        simdgroup_mlp8x8 output[OutputDim/8];
        simdgroup_mlp8x8 hidden[HiddenLayers][HiddenNeurons/8];
        simdgroup_mlp8x8 input[InputDim/8];
#ifdef SHARED_MEMORY_WEIGHTS
        const threadgroup FloatMLP* cur_weights = smem_weights
#else
        const device FloatMLP* cur_weights = weights
#endif
        + (
            InputDim * HiddenNeurons +
            (HiddenLayers - 1) * HiddenNeurons * HiddenNeurons +
            HiddenNeurons * OutputDim
        );
        
        load(output, dL_dactivations + OutputDim * index + batch_size * (InputDim + HiddenLayers * HiddenNeurons));

        backward(hidden[HiddenLayers - 1], output, cur_weights);

        unrolled for (uint minus = 1; minus < HiddenLayers; minus++) {
            const uint layer = HiddenLayers - minus; // weird trick to get clang to unroll this loop
            activation_gradient(hidden[layer], activations + HiddenNeurons * index + batch_size * (InputDim + layer * HiddenNeurons));
            if (StoreActivations) {
                store(hidden[layer], dL_dactivations + HiddenNeurons * index + batch_size * (InputDim + layer * HiddenNeurons));
            }
            backward(hidden[layer - 1], hidden[layer], cur_weights);
        }
        activation_gradient(hidden[0], activations + HiddenNeurons * index + batch_size * InputDim);
        if (StoreActivations) {
            store(hidden[0], dL_dactivations + HiddenNeurons * index + batch_size * InputDim);
        }
        backward(input, hidden[0], cur_weights);
        
        store(input, dL_dactivations + InputDim * index);
        
        index += 8;
    }
}

template<uint PrevDim, uint NextDim>
void mlp_backward_weights_impl(
    constant uint& batch_size,
    constant uint& layer,
    
    const device FloatMLP* activations,
    const device FloatMLP* dL_dactivations,
    device atomic<FloatFP>* dL_dweights,
    
    threadgroup FloatMLP* buffer,
    
    uint2 threadIdx,
    uint2 gridDim,
    ushort laneId
) {
    constexpr uint PrevTiles = PrevDim / 8;
    constexpr uint NextTiles = NextDim / 8;
    
    constexpr uint PrevTilesSliced = PrevTiles * NextTiles > (192 / sizeof(FloatMLP)) ? 8 : PrevTiles;
    constexpr uint NextTilesSliced = PrevTiles * NextTiles > (192 / sizeof(FloatMLP)) ? 4 : NextTiles;
    static_assert((PrevTiles % PrevTilesSliced) == 0, "slicing failed");
    static_assert((NextTiles % NextTilesSliced) == 0, "slicing failed");
    
    dL_dactivations += batch_size * (InputDim + layer * HiddenNeurons); // next layer
    if (layer > 0) {
        activations += batch_size * (InputDim + (layer - 1) * HiddenNeurons); // prev layer
        dL_dweights += InputDim * HiddenNeurons + (layer - 1) * HiddenNeurons * HiddenNeurons;
    }
    
    unrolled for (uint prevSlice = 0; prevSlice < PrevTiles; prevSlice += PrevTilesSliced) {
        unrolled for (uint nextSlice = 0; nextSlice < NextTiles; nextSlice += NextTilesSliced) {
            // TODO: accumulate in full precision?
            simdgroup_mlp8x8 weights_grad[PrevTilesSliced][NextTilesSliced]; // prev_dim x next_dim
            unrolled for (uint prev = 0; prev < PrevTilesSliced; prev++) {
                unrolled for (uint next = 0; next < NextTilesSliced; next++) {
                    weights_grad[prev][next] = make_filled_simdgroup_matrix<FloatMLP, 8, 8>(0);
                }
            }
            
            uint index = 8 * (threadIdx.x / 32);
            for (; index < batch_size; index += 8 * (gridDim.x / 32)) {
                simdgroup_mlp8x8 activation[PrevTilesSliced]; // prev_dim x batch
                simdgroup_mlp8x8 dL_dactivation[NextTilesSliced]; // batch x next_dim
                
                unrolled for (uint prev = 0; prev < PrevTilesSliced; prev++) {
                    simdgroup_load(activation[prev], activations
                        + index * PrevDim
                        + 8 * (prev + prevSlice), PrevDim, ulong2(0, 0), true);
                }
                unrolled for (uint next = 0; next < NextTilesSliced; next++) {
                    simdgroup_load(dL_dactivation[next], dL_dactivations
                        + index * NextDim
                        + 8 * (next + nextSlice), NextDim);
                }
                
                unrolled for (uint prev = 0; prev < PrevTilesSliced; prev++) {
                    unrolled for (uint next = 0; next < NextTilesSliced; next++) {
                        simdgroup_multiply_accumulate(
                            weights_grad[prev][next], activation[prev], dL_dactivation[next], weights_grad[prev][next]);
                    }
                }
            }
            
            unrolled for (uint prev = 0; prev < PrevTilesSliced; prev++) {
                unrolled for (uint next = 0; next < NextTilesSliced; next++) {
                    simdgroup_store(weights_grad[prev][next], buffer, 8, ulong2(0, 0), true);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    
                    device atomic<FloatFP>* frag = dL_dweights +
                        (8 * (prev + prevSlice) + laneId % 8) +
                        (8 * (next + nextSlice) + laneId / 8) * PrevDim;
                    atomic_fetch_add_explicit(frag, FloatFP(buffer[laneId]), memory_order_relaxed);
                    atomic_fetch_add_explicit(frag + 4 * PrevDim, FloatFP(buffer[laneId + 32]), memory_order_relaxed);
                }
            }
        }
    }
}

kernel void mlp_optimize_adam(
    constant uint& current_step,

    constant float& learning_rate,
    constant float& l2_regularization,
    
    const device FloatFP* dL_dweights,
    device FloatFP* weights_fp,
    device FloatMLP* weights,
    
    device FloatFP* first_moments,
    device FloatFP* second_moments,
    
    uint i [[thread_position_in_grid]]
) {
    device FloatFP& weight = weights_fp[i];
    const FloatFP gradient = dL_dweights[i] + l2_regularization * weight;
    
    //if (dL_dweights[i] == 0) return;
    
    //weight -= learning_rate * gradient;
    //return;
    
    constexpr FloatFP beta1 = 0.9f;
    constexpr FloatFP beta2 = 0.999f;
    constexpr FloatFP epsilon = 1.e-8;
    
    const FloatFP gradient_sq = gradient * gradient;
    
    const FloatFP first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
    const FloatFP second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;
    
    const FloatFP bias_correction_1 = 1 - pow(beta1, FloatFP(current_step));
    const FloatFP bias_correction_2 = 1 - pow(beta2, FloatFP(current_step));
    const FloatFP lr = learning_rate / bias_correction_1;
    FloatFP effective_learning_rate = lr / (sqrt(second_moment) / sqrt(bias_correction_2) + epsilon);

    weight -= effective_learning_rate * first_moment;
    weights[i] = FloatMLP(weight);
}

kernel void adam_copy_weights(
    device FloatMLP* weights,
    device FloatFP* weights_fp,
    uint i [[thread_position_in_grid]]
) {
    weights_fp[i] = FloatFP(weights[i]);
}

kernel void mlp_random_weights(
    constant uint& seed,
    constant float& scale,

    device FloatMLP* weights,
    
    uint i [[thread_position_in_grid]]
) {
    float weight = scale * (2 * sample_tea_float32(seed, i) - 1);
    weights[i] = FloatMLP(weight);
}

#define make_mlp_backward(name, PrevDim, NextDim) \
kernel void name(\
    constant uint& batch_size, \
    constant uint& layer, \
     \
    const device FloatMLP* activations, \
    const device FloatMLP* dL_dactivations, \
    device atomic<FloatFP>* dL_dweights, \
     \
    uint2 threadIdx [[thread_position_in_grid]], \
    uint2 gridDim [[threads_per_grid]], \
    ushort laneId [[thread_index_in_simdgroup]] \
) { \
    threadgroup FloatMLP buffer[8 * 8]; \
    mlp_backward_weights_impl<PrevDim, NextDim>( \
        batch_size, layer, activations, dL_dactivations, dL_dweights, buffer, \
        threadIdx, gridDim, laneId); \
}

make_mlp_backward(mlp_backward_weights_input, InputDim, HiddenNeurons)
make_mlp_backward(mlp_backward_weights_hidden, HiddenNeurons, HiddenNeurons)
make_mlp_backward(mlp_backward_weights_output, HiddenNeurons, OutputDim)
