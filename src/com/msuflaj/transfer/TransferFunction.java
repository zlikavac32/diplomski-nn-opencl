package com.msuflaj.transfer;

public interface TransferFunction {

    public double calculate(double value);

    public double derivative(double value);

    public int getId();

    public double[] getParams();

}
