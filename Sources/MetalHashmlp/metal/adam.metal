#include <metal_stdlib>

using namespace metal;

kernel void step(
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

kernel void copy(
    device FloatMLP* weights,
    device FloatFP* weights_fp,
    uint i [[thread_position_in_grid]]
) {
    weights_fp[i] = FloatFP(weights[i]);
}
