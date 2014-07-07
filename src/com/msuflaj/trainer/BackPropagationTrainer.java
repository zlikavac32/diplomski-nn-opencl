package com.msuflaj.trainer;

import java.util.Random;

public class BackPropagationTrainer extends PropagationTrainer {

    private class ErrorNeuron extends PropagationTrainer.ErrorNeuron {

        public ErrorNeuron(PropagationCompatibleNeuron neuron, PropagationTrainer.ErrorNeuron[] frontNeurons, int index) {
            super(neuron, frontNeurons, index);
        }

        @Override
        protected void updateWeightsSpecific(double ni) {

            double[] weights = neuron.getWeights();

            for (int i = 0; i < weights.length; i++) {
                weights[i] += ni * gradients[i];
            }

        }

    }

    public BackPropagationTrainer(double ni, double lo, double hi, Random random) {
        super(ni, lo, hi, random);
    }

    @Override
    protected PropagationTrainer.ErrorNeuron createErrorNeuron(PropagationCompatibleNeuron regularNeuron, PropagationTrainer.ErrorNeuron[] errorNeurons, int index) {
        return new ErrorNeuron(regularNeuron, errorNeurons, index);
    }

    @Override
    public boolean supportsOnlineMode() {
        return true;
    }
}
