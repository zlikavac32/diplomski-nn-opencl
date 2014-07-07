package com.msuflaj.transfer;

public class SigmoidFunction implements TransferFunction {

    @Override
    public double calculate(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    @Override
    public double derivative(double value) {
        double v = calculate(value);
        return (1 - v) * v;
    }

    @Override
    public int getId() {
        return 2;
    }

    @Override
    public double[] getParams() {
        return new double[0];
    }
}
