package com.msuflaj.network;

import com.msuflaj.trainer.PropagationTrainer;
import com.msuflaj.transfer.LinearFunction;
import com.msuflaj.transfer.TransferFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class FullyConnectedForwardNetwork implements PropagationTrainer.Trainable {

    protected final int inputNeuronsCount;

    protected final int outputNeuronsCount;

    protected int weightsOffset;

    protected int[] dimensions;

    protected List<Neuron[]> neurons;

    protected TransferFunction[] transferFunctions;

    protected boolean isBiased;

    public static class BiasedNeuron implements PropagationTrainer.PropagationCompatibleNeuron {

        private static TransferFunction function = new LinearFunction();

        @Override
        public double getOutput() {
            return 1;
        }

        @Override
        public double[] getWeights() {
            return new double[0];
        }

        @Override
        public double getRawOutput() {
            return 1;
        }

        @Override
        public TransferFunction getTransferFunction() {
            return function;
        }

        @Override
        public Neuron[] getConnections() {
            return new Neuron[0];
        }
    }

    public static class InputNeuron implements Neuron {

        protected double input;

        public void setInput(double input) {
            this.input = input;
        }

        public double getOutput() {
            return input;
        }

    }

    public static class RegularNeuron implements PropagationTrainer.PropagationCompatibleNeuron {

        protected double output;

        protected double rawOutput;

        protected double[] weights;

        protected Neuron[] neurons;

        protected TransferFunction transferFunction;

        public void connect(Neuron[] neurons) {
            if (null != weights && neurons.length != weights.length) {
                throw new IllegalArgumentException("Neurons and weights arrays should have same dimensions");
            }
            this.neurons = neurons;
        }

        @Override
        public Neuron[] getConnections() {
            return neurons;
        }

        public void setWeights(double[] weights) {
            if (null != neurons) {
                if (neurons.length != weights.length) {
                    throw new IllegalArgumentException("Neurons and weights arrays should have same dimensions");
                }
            }
            this.weights = weights.clone();
        }

        @Override
        public double[] getWeights() {
            return weights;
        }

        public void calculate() {
            rawOutput = 0;
            for (int i = 0 ; i < neurons.length; i++) {
                rawOutput += neurons[i].getOutput() * weights[i];
            }
            output = transferFunction.calculate(rawOutput);
        }

        @Override
        public double getOutput() {
            return output;
        }

        @Override
        public double getRawOutput() {
            return rawOutput;
        }

        @Override
        public TransferFunction getTransferFunction() {
            return transferFunction;
        }

        public void setTransferFunction(TransferFunction function) {
            transferFunction = function;
        }

    }

    public FullyConnectedForwardNetwork(int[] dimensions, TransferFunction[] functions) throws NetworkException {
        this(dimensions, functions, true);
    }

    public FullyConnectedForwardNetwork(int[] dimensions, TransferFunction[] functions, boolean isBiased) throws NetworkException {
        if (dimensions.length < 2) {
            throw new NetworkException("Layer count can not be less than two");
        }

        if (dimensions.length - 1 != functions.length) {
            throw new NetworkException("Input neurons do not need transfer function (only hidden and output layers)");
        }

        neurons = new ArrayList<>(dimensions.length);

        if (dimensions[0]< 1) {
            throw new NetworkException("Input neurons count must be greater than 0");
        }

        int extra = 0;

        if (isBiased) { extra = 1; }

        Neuron[] neurons = new Neuron[dimensions[0] + extra];

        for (int i = 0; i < dimensions[0]; i++) {
            neurons[i] = new InputNeuron();
        }

        if (isBiased) {
            neurons[dimensions[0]] = new BiasedNeuron();
        }

        weightsOffset = 0;

        this.neurons.add(neurons);

        for (int i = 1; i < dimensions.length - 1; i++) {
            int d = dimensions[i];

            if (d < 1) {
                throw new NetworkException("Number of neurons in hidden layer " + i + " can not be less than 0");
            }

            weightsOffset += dimensions[i] * (dimensions[i - 1] + extra);

            Neuron[] newLayer = new Neuron[d + extra];
            TransferFunction function = functions[i - 1];

            for (int j = 0; j < d; j++) {
                RegularNeuron neuron = new RegularNeuron();
                neuron.connect(neurons);
                neuron.setWeights(new double[neurons.length]);
                neuron.setTransferFunction(function);
                newLayer[j] = neuron;
            }

            if (isBiased) {
                newLayer[d] = new BiasedNeuron();
            }

            neurons = newLayer;

            this.neurons.add(newLayer);
        }

        int d = dimensions[dimensions.length - 1];

        if (d < 1) {
            throw new NetworkException("Output layer neurons count must be greater than 0");
        }


        weightsOffset += d * (dimensions[dimensions.length - 2] + extra);

        Neuron[] newLayer = new Neuron[d];

        TransferFunction function = functions[functions.length - 1];

        for (int i = 0; i < d; i++) {
            RegularNeuron neuron = new RegularNeuron();
            neuron.connect(neurons);
            neuron.setWeights(new double[neurons.length]);
            neuron.setTransferFunction(function);
            newLayer[i] = neuron;
        }

        inputNeuronsCount = dimensions[0];
        outputNeuronsCount = d;

        this.neurons.add(newLayer);

        this.isBiased = isBiased;
        this.dimensions =  dimensions.clone();
        this.neurons = Collections.unmodifiableList(this.neurons);
        this.transferFunctions = functions.clone();

    }

    public int getOutputNeuronsCount() {
        return outputNeuronsCount;
    }

    public List<Neuron[]> getNeurons() {
        return neurons;
    }

    @Override
    public boolean isBiased() {
        return isBiased;
    }

    @Override
    public TransferFunction[] getTransferFunctions() {
        return transferFunctions;
    }

    @Override
    public int[] getDimensions() {
        return dimensions;
    }

    @Override
    public double[] getWeights() {
        double[] weights = new double[weightsOffset];

        int extra = isBiased ? 1 : 0;

        int offset = 0;

        for (int i = 1; i < dimensions.length; i++) {
            Neuron[] neuronsList = neurons.get(i);

            for (int j = 0, limit = neuronsList.length - (((i + 1) < dimensions.length) ? extra : 0); j < limit; j++) {
                RegularNeuron neuron = (RegularNeuron) neuronsList[j];
                double[] tempWeights = neuron.getWeights();
                System.arraycopy(tempWeights, 0, weights, offset, tempWeights.length);
                offset += tempWeights.length;
            }

        }

        return weights;
    }

    @Override
    public int getWeightsCount() {
        return weightsOffset;
    }

    @Override
    public void setWeights(double[] weights) {

        int extra = isBiased ? 1 : 0;

        int offset = 0;

        for (int i = 1; i < dimensions.length; i++) {
            Neuron[] neuronsList = neurons.get(i);

            double[] tempWeights = new double[dimensions[i - 1] + extra];

            for (int j = 0, limit = neuronsList.length - (((i + 1) < dimensions.length) ? extra : 0); j < limit; j++) {
                RegularNeuron neuron = (RegularNeuron) neuronsList[j];
                System.arraycopy(weights, offset, tempWeights, 0, tempWeights.length);
                neuron.setWeights(tempWeights);
                offset += tempWeights.length;
            }

        }
    }

    @Override
    public double[] process(double[] values) throws NetworkException {
        double[] output = new double[getOutputNeuronsCount()];
        process(values, output);
        return output;
    }

    @Override
    public void process(double[] values, double[] ret) throws NetworkException {

        Neuron[] inputNeurons = neurons.get(0);

        int extra = (isBiased) ? 1 : 0;

        for (int i = 0; i < inputNeuronsCount; i++) {
            ((InputNeuron) inputNeurons[i]).setInput(values[i]);
        }

        for (int i = 1; i < dimensions.length - 1; i++) {
            Neuron[] neurons = this.neurons.get(i);

            for (int j = 0, l = neurons.length - extra; j < l; j++) {
                ((RegularNeuron) neurons[j]).calculate();
            }

        }

        Neuron[] outputNeurons = neurons.get(dimensions.length - 1);

        for (int i = 0; i < outputNeurons.length; i++) {
            ((RegularNeuron) outputNeurons[i]).calculate();
            ret[i] = outputNeurons[i].getOutput();
        }

    }
}
