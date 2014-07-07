package com.msuflaj.trainer;

import com.msuflaj.dataset.DataSet;
import com.msuflaj.network.Network;
import com.msuflaj.network.NetworkException;
import com.msuflaj.network.Neuron;
import com.msuflaj.statistics.Statistics;
import com.msuflaj.transfer.TransferFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public abstract class PropagationTrainer implements Trainer {

    private final double lo;

    private final double hi;

    private final Random random;

    private double ni;

    private boolean isOnlineMode = false;
    
    public static interface Trainable extends Network {

        public int getOutputNeuronsCount();

        public List<Neuron[]> getNeurons();

        public boolean isBiased();

    }
    
    public static interface PropagationCompatibleNeuron extends Neuron {

        public double[] getWeights();

        public double getRawOutput();

        public TransferFunction getTransferFunction();

        public Neuron[] getConnections();

    }

    protected abstract class ErrorNeuron {

        protected final ErrorNeuron[] frontNeurons;

        protected final PropagationCompatibleNeuron neuron;

        protected final int index;

        protected double[] gradients;

        protected double[] weights;

        protected double error;

        public ErrorNeuron(PropagationCompatibleNeuron neuron, ErrorNeuron[] frontNeurons, int index) {
            this.neuron = neuron;
            this.frontNeurons = frontNeurons;
            this.index = index;
            weights = neuron.getWeights().clone();
            gradients = new double[weights.length];
        }

        /**
         * If called multiple times should accumulate gradient
         *
         * @param expected Expected error
         */
        public void calculateError(double expected) {
            error = neuron.getTransferFunction().derivative(neuron.getRawOutput()) * (expected - neuron.getOutput());
        }

        /**
         * If called multiple times should accumulate gradient
         */
        public void calculateError() {
            double sum = 0;

            for (ErrorNeuron neuron : frontNeurons) {
                sum += neuron.error * neuron.neuron.getWeights()[index];
            }

            error = neuron.getTransferFunction().derivative(neuron.getRawOutput()) * sum;
        }

        public void calculateGradients() {
            Neuron[] connections = neuron.getConnections();

            for (int i = 0; i < connections.length; i++) {
                gradients[i] += connections[i].getOutput() * error;
            }
        }

        public void storeWeights() {
            System.arraycopy(neuron.getWeights(), 0, weights, 0, weights.length);
        }

        public void restoreWeights() {
            System.arraycopy(weights, 0, neuron.getWeights(), 0, weights.length);
        }

        public void updateWeights(double ni) {
            updateWeightsSpecific(ni);
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] = 0;
            }
        }

        /**
         * When called should delete calculated errors
         * @param ni Ni coefficient
         */
        protected abstract void updateWeightsSpecific(double ni);

    }

    public PropagationTrainer(double ni, double lo, double hi, Random random) {
        if (ni < 0) {
            throw new IllegalArgumentException("Ni must be positive");
        }

        if (lo >= hi) {
            throw new IllegalArgumentException("Down boundary must be less than upper boundary");
        }

        this.ni = ni;
        this.lo = lo;
        this.hi = hi;
        this.random = random;
    }

    @Override
    public double train(Network net, DataSet dataSet, StopCondition stopCondition, Statistics statistics) throws NetworkException, UnexpectedNetworkException {

        if (!(net instanceof Trainable)) {
            throw new UnexpectedNetworkException("Expected network must implement " + Trainable.class);
        }

        Trainable network = (Trainable) net;

        List<Neuron[]> listOfNeurons = network.getNeurons();
        List<ErrorNeuron[]> listOfErrorNeurons = new ArrayList<>(listOfNeurons.size() - 1);

        boolean isBiased = network.isBiased();
        ErrorNeuron[] lastErrorNeurons = null;

        double range = hi - lo;

        for (int i = listOfNeurons.size() - 1, l = i; i > 0; i--) {

            Neuron[] neurons = listOfNeurons.get(i);

            ErrorNeuron[] errorNeurons = new ErrorNeuron[neurons.length];

            for (int j = 0; j < errorNeurons.length; j++) {
                PropagationCompatibleNeuron neuron = (PropagationCompatibleNeuron) neurons[j];
                double[] weights = neuron.getWeights();
                for (int k = 0; k < weights.length; k++) {
                    weights[k] = random.nextDouble() * range + lo;
                }
                errorNeurons[j] = createErrorNeuron(neuron, lastErrorNeurons, j);

            }

            listOfErrorNeurons.add(errorNeurons);
            if (isBiased && i < l) {
                errorNeurons = Arrays.copyOf(errorNeurons, errorNeurons.length - 1);
            }
            lastErrorNeurons = errorNeurons;

        }

        double[] values = new double[network.getOutputNeuronsCount()];

        double error = calculateError(network, dataSet);
        double bestError = error;

        double[][] in = dataSet.first;
        double[][] out = dataSet.second;

        statistics.signalStart();
        statistics.setError(error);

        for (int i = 0; stopCondition.isConditionMet(i, error); i++) {

            for (int u = 0; u < in.length; u++) {

                network.process(in[u], values);

                double[] expected = out[u];

                ErrorNeuron[] errorNeurons = listOfErrorNeurons.get(0);

                for (int k = 0; k < errorNeurons.length; k++) {
                    errorNeurons[k].calculateError(expected[k]);
                }

                for (int j = 1, l = listOfErrorNeurons.size(); j < l; j++) {

                    errorNeurons = listOfErrorNeurons.get(j);

                    for (ErrorNeuron errorNeuron : errorNeurons) {
                        errorNeuron.calculateError();
                    }
                }

                for (ErrorNeuron[] listOfErrorNeuron : listOfErrorNeurons) {
                    for (ErrorNeuron neuron : listOfErrorNeuron) {
                        neuron.calculateGradients();
                        if (isOnlineMode) {
                            neuron.updateWeights(ni);
                        }
                    }
                }
            }

            if (!isOnlineMode) {
                for (ErrorNeuron[] listOfErrorNeuron : listOfErrorNeurons) {
                    for (ErrorNeuron neuron : listOfErrorNeuron) {
                        neuron.updateWeights(ni);
                    }
                }
            }

            error = calculateError(network, dataSet);

            if (error < bestError) {
                for (ErrorNeuron[] errorNeurons : listOfErrorNeurons) {
                    for (ErrorNeuron errorNeuron : errorNeurons) {
                        errorNeuron.storeWeights();
                    }
                }
                bestError = error;
            }

            statistics.incrementIteration();
            statistics.setError(error);

        }

        statistics.signalFinish();

        for (ErrorNeuron[] errorNeurons : listOfErrorNeurons) {
            for (ErrorNeuron errorNeuron : errorNeurons) {
                errorNeuron.restoreWeights();
            }
        }

        return bestError;
    }

    private double calculateError(Trainable network, DataSet dataSet) throws NetworkException {
        double[] values = new double[network.getOutputNeuronsCount()];

        double error = 0;

        for (int i = 0; i < dataSet.first.length; i++) {
            network.process(dataSet.first[i], values);
            double[] temp = dataSet.second[i];
            for (int j = 0; j < values.length; j++) {
                error += (values[j] - temp[j]) * (values[j] - temp[j]);
            }
        }

        return error / (dataSet.first.length << 1);
    }

    public void setOnlineMode(boolean onlineMode) throws NetworkException {
        if (onlineMode && !supportsOnlineMode()) {
            throw new NetworkException("Implementation does not support online mode");
        }
        this.isOnlineMode = onlineMode;
    }

    protected abstract ErrorNeuron createErrorNeuron(PropagationCompatibleNeuron regularNeuron, ErrorNeuron[] errorNeurons, int index);

    public abstract boolean supportsOnlineMode();

}
