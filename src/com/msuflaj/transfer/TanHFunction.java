package com.msuflaj.transfer;

public class TanHFunction implements TransferFunction {

    @Override
    public double calculate(double value) {
        return Math.tanh(value);
    }

    @Override
    public double derivative(double value) {
        double v = calculate(value);
        return 1 - v * v;
    }

    @Override
    public int getId() {
        return 1;
    }

    @Override
    public double[] getParams() {
        return new double[0];
    }
}
