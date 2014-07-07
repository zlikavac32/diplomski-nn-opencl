float transferFunctionLinear(const float value, const float param) {
    return value * param;
}

float transferFunctionSigmoid(const float value) {
    return 1 / (1 + exp(-value));
}

float transferFunctionTanH(const float value) {
    return tanh(value);
}

float computeTransfer(const float raw, const int transferFunctionId, global const float *params) {
    switch (transferFunctionId) {
        case 1 :
            return transferFunctionTanH(raw);
        case 2 :
            return transferFunctionSigmoid(raw);
        case 3 :
            return transferFunctionLinear(raw, params[0]);
        default :
            return 0;
    }

}

float transferFunctionLinearDerivative(const float value, const float param) {
    return param;
}

float transferFunctionSigmoidDerivative(const float value) {
    float v = transferFunctionSigmoid(value);
    return (1 - v) * v;
}

float transferFunctionTanHDerivative(const float value) {
    float v = transferFunctionTanH(value);
    return 1 - v * v;
}

float computeTransferDerivative(const float raw, const int transferFunctionId, global const float *params) {

    switch (transferFunctionId) {
        case 1 :
            return transferFunctionTanHDerivative(raw);
        case 2 :
            return transferFunctionSigmoidDerivative(raw);
        case 3 :
            return transferFunctionLinearDerivative(raw, params[0]);
        default :
            return 0;
    }

}

/**
 * Weights are ordered in [w01, w02, w03..., wij] where i is neuron index in layer
 * and j is neuron index in layer - 1 (indexes are 0-based)
 * @var isBiased 0 if network is not biased, 1 otherwise
 * @var number of layers in network
 * @var total number of neurons
 * @var neuronOffsets Pre-computed starting indexes for neuron values
 * @var weightsOffsets Pre-computed starting indexes for weights values
 * @var layerDimensions Number of neurons in each layer (without biased neuron)
 * @var transferFunctions ID of each transfer function in layer
 * @var transferFunctionParams All transfer function params that will be used
 * @var transferFunctionParamsOffsets For each transfer function offset for it's params in respect to previous layer
 * @var dataSets array with all elements to evaluate
 * @var weights Array with all weights
 * @var rawOutputs Array with all neuron outputs without transfer function applied
 * @var outputs Array with all neuron outputs with transfer function applied
 */
kernel void evaluate(
    const int isBiased,
    const int numberOfLayers,
    const int numberOfNeurons,
    const int calculateError,
    global const int *neuronOffsets,
    global const int *weightsOffsets,
    global const int *layerDimensions,
    global const int *transferFunctions,
    global const float *transferFunctionParams,
    global const int *transferFunctionParamsOffsets,
    global const float *dataSets,
    global const float *weights,
    global float *rawOutputs,
    global float *outputs,
    global float *errors
) {

    int id = get_global_id(0);

    int outputOffset = id * numberOfNeurons;
    int dataSetOffset = id * (layerDimensions[0] + layerDimensions[numberOfLayers - 1]);

    for (int i = 0; i < layerDimensions[0]; i++) {
        outputs[outputOffset + i] = dataSets[dataSetOffset + i];
        rawOutputs[outputOffset + i] = dataSets[dataSetOffset + i];
    }

    //For each layer
    for (int i = 1; i < numberOfLayers; i++) {
        //For each neuron in current layer
        for (int j = 0; j < layerDimensions[i]; j++) {

            int size = layerDimensions[i - 1] + isBiased;
            int initialNeuronOffset = neuronOffsets[i - 1] + outputOffset;
            int myLevelNeuronOffset = neuronOffsets[i] + outputOffset;

            //Given this layer weights offset, skip weights that were before me
            int initialWeightsOffset = weightsOffsets[i - 1] +
                (layerDimensions[i - 1] + isBiased) * j;

            float sum = 0;

            for (int k = 0; k < size; k++) {

                sum += weights[initialWeightsOffset + k] * outputs[initialNeuronOffset + k];
            }

            rawOutputs[myLevelNeuronOffset + j] = sum;

            float ret = computeTransfer(
                sum, transferFunctions[i - 1],
                transferFunctionParams + transferFunctionParamsOffsets[i - 1]
            );

            outputs[myLevelNeuronOffset + j] = ret;
        }
    }

    if (calculateError) {
        int startOffset = outputOffset + numberOfNeurons - layerDimensions[numberOfLayers - 1];
        float sum = 0;
        dataSetOffset += layerDimensions[0];
        for (int i = 0, outputNeuronsCount = layerDimensions[numberOfLayers - 1]; i < outputNeuronsCount; i++) {
            sum += (outputs[startOffset + i] - dataSets[dataSetOffset + i]) * (outputs[startOffset + i] - dataSets[dataSetOffset + i]);
        }
        errors[id] = sum;
    }
}

kernel void calculateLayerError(
    const int isBiased,
    const int numberOfLayers,
    const int numberOfNeurons,
    global const int *neuronOffsets,
    global const int *weightsOffsets,
    global const int *layerDimensions,
    global const int *transferFunctions,
    global const float *transferFunctionParams,
    global const int *transferFunctionParamsOffsets,
    global const float *dataSets,
    global const float *weights,
    global const float *rawOutputs,
    global const float *outputs,
    global float *neuronErrors
) {

    int id = get_global_id(0);

    int layerId = numberOfLayers - 1;
    int outputOffsetStart = id * numberOfNeurons + neuronOffsets[layerId];
    int dataSetOffset = id * (layerDimensions[0] + layerDimensions[layerId]) + layerDimensions[0];

    for (int i = 0, limit = layerDimensions[numberOfLayers - 1]; i < limit; i++) {
        neuronErrors[outputOffsetStart + i] = computeTransferDerivative(
            rawOutputs[outputOffsetStart + i], transferFunctions[layerId - 1],
            transferFunctionParams + transferFunctionParamsOffsets[layerId - 1]
        ) * (dataSets[dataSetOffset + i] - outputs[outputOffsetStart + i]);
    }

    //For each hidden layer
    for (int i = layerId - 1; i > 0; i--) {
        int myLevelOutputOffsetStart = id * numberOfNeurons + neuronOffsets[i];
        int nextLevelOutputOffsetStart = id * numberOfNeurons + neuronOffsets[i + 1];

        //For each neuron in my layer
        for (int j = 0, currentDimensions = layerDimensions[i] + isBiased; j < layerDimensions[i]; j++) {


            float sum = 0;

            //For each neuron in outer layer
            for (int k = 0, limit = layerDimensions[i + 1]; k < limit; k++) {
                sum += neuronErrors[nextLevelOutputOffsetStart + k] * weights[weightsOffsets[i] + k * currentDimensions + j];
            }

            neuronErrors[myLevelOutputOffsetStart + j] = computeTransferDerivative(
                rawOutputs[myLevelOutputOffsetStart + j], transferFunctions[i - 1],
                transferFunctionParams + transferFunctionParamsOffsets[i - 1]
            ) * sum;
        }

    }

    if (isBiased) {
        for (int i = layerId - 1; i > 0; i--) {
            int myLevelOutputOffsetStart = id * numberOfNeurons + neuronOffsets[i];
            int nextLevelOutputOffsetStart = id * numberOfNeurons + neuronOffsets[i + 1];

            int j = layerDimensions[i];

            float sum = 0;

            //For each neuron in outer layer
            for (int k = 0, limit = layerDimensions[i + 1]; k < limit; k++) {
                sum += neuronErrors[nextLevelOutputOffsetStart + k] * weights[weightsOffsets[i] + k * (j + 1) + j];
            }

            neuronErrors[myLevelOutputOffsetStart + j] = sum;

        }
    }

}


kernel void calculateGradients(
    const int isBiased,
    const int numberOfLayers,
    const int numberOfNeurons,
    const int numberOfWeights,
    global const int *neuronOffsets,
    global const int *weightsOffsets,
    global const int *layerDimensions,
    global const float *outputs,
    global const float *neuronErrors,
    global float *gradients
) {

    int id = get_global_id(0);

    int initialNeuronOffset = numberOfNeurons * id;
    int initialWeightsOffset = numberOfWeights * id;

    for (int i = 1; i < numberOfLayers; i++) {

        for (int j = 0, limit = layerDimensions[i]; j < limit; j++) {

            float error = neuronErrors[initialNeuronOffset + neuronOffsets[i] + j];

            //Given this layer weights offset, skip weights that were before me
            int currentWeightsOffset = initialWeightsOffset + weightsOffsets[i - 1] +
                            (layerDimensions[i - 1] + isBiased) * j;
            int currentNeuronOffset = initialNeuronOffset + neuronOffsets[i - 1];

            for (int k = 0; k < layerDimensions[i - 1] + isBiased; k++) {
                gradients[currentWeightsOffset + k] = error * outputs[currentNeuronOffset + k];
            }

        }

    }

}

kernel void sumErrors(
    const int size,
    const global float *errors,
    global float *totalError
) {

    float sum = 0;

    for (int i = 0; i < size; i++) {
        sum += errors[i];
    }

    totalError[0] = sum / (size << 1);

}

kernel void sumGradients(
    const int numberOfWeights,
    const int dataSetSize,
    global float *gradients
) {
    int id = get_global_id(0);
    for (int i = 1, index = numberOfWeights; i < dataSetSize; i++, index += numberOfWeights) {
        gradients[id] += gradients[id + index];
        gradients[id + index] = 0;
    }
}
