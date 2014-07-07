
package com.msuflaj.trainer;

import java.util.Random;

public class RPROPTrainer extends PropagationTrainer {

    private double deltaInitial;

    private double deltaMin;

    private double deltaMax;

    private double kMinus;

    private double kPlus;

    private class ErrorNeuron extends PropagationTrainer.ErrorNeuron {

        private double[] previousGradients;

        private double[] deltas;

        public ErrorNeuron(PropagationCompatibleNeuron neuron, PropagationTrainer.ErrorNeuron[] frontNeurons, int index) {
            super(neuron, frontNeurons, index);
            previousGradients = new double[gradients.length];
            deltas = new double[previousGradients.length];

            for (int i = 0; i < deltas.length; i++) {
                deltas[i] = deltaInitial;
            }
        }

        @Override
        public void updateWeightsSpecific(double ni) {

            double[] weights = neuron.getWeights();

            for (int i = 0; i < weights.length; i++) {

                double currentGradient = -gradients[i];

                double gradientsMultiplied = currentGradient * previousGradients[i];

                previousGradients[i] = currentGradient;

                double currentDelta = deltas[i];

                if (gradientsMultiplied > 0) {
                    currentDelta *= kPlus;
                } else if (gradientsMultiplied < 0) {
                    currentDelta *= kMinus;
                }

                if (currentDelta > deltaMax) {
                    currentDelta = deltaMax;
                } else if (currentDelta < deltaMin) {
                    currentDelta = deltaMin;
                }

                deltas[i] = currentDelta;

                if (currentGradient > 0) {
                    weights[i] -= currentDelta;
                } else if (currentGradient < 0) {
                    weights[i] += currentDelta;
                }

            }

        }

    }

    public RPROPTrainer(double kMinus, double kPlus, double deltaInitial, double deltaMin, double deltaMax, double lo, double hi, Random random) {
        super(0, lo, hi, random);
        this.kMinus = kMinus;
        this.kPlus = kPlus;
        this.deltaMin = deltaMin;
        this.deltaMax = deltaMax;
        this.deltaInitial = deltaInitial;
    }

    @Override
    protected PropagationTrainer.ErrorNeuron createErrorNeuron(PropagationCompatibleNeuron regularNeuron, PropagationTrainer.ErrorNeuron[] errorNeurons, int index) {
        return new ErrorNeuron(regularNeuron, errorNeurons, index);
    }

    @Override
    public boolean supportsOnlineMode() {
        return false;
    }
}
