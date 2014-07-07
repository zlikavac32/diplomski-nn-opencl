
kernel void updateWeights(
    const float kPlus,
    const float kMinus,
    const float deltaMax,
    const float deltaMin,
    global float *gradients,
    global float *previousGradients,
    global float *deltas,
    global float *weights
) {
    int id = get_global_id(0);

    float currentGradient = -gradients[id];

    float gradientsMultiplied = currentGradient * previousGradients[id];

    previousGradients[id] = currentGradient;

    float currentDelta = deltas[id];

    currentDelta *= (gradientsMultiplied > 0) * kPlus + (gradientsMultiplied < 0) * kMinus + (0 == gradientsMultiplied);

    currentDelta = (currentDelta > deltaMax) * deltaMax + (currentDelta < deltaMin) * deltaMin + (currentDelta <= deltaMax && currentDelta >= deltaMin) * currentDelta;

    deltas[id] = currentDelta;

    weights[id] += (-(currentGradient > 0) + (currentGradient < 0)) * currentDelta;

    gradients[id] = 0;

}