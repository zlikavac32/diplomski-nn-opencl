package com.msuflaj.transfer;

public class LinearFunction implements TransferFunction {

    private final double mult;

    public LinearFunction() {
        this(1);
    }

    public LinearFunction(double mult) {
        this.mult = mult;
    }

    @Override
    public double calculate(double value) {
        return value * mult;
    }

    @Override
    public double derivative(double value) {
        return mult;
    }

    @Override
    public int getId() {
        return 3;
    }

    @Override
    public double[] getParams() {
        return new double[] {
            mult
        };
    }
}
