
kernel void updateWeights(
    const float ni,
    global float *gradients,
    global float *weights
) {
    int id = get_global_id(0);
    weights[id] += ni * gradients[id];
    gradients[id] = 0;
}