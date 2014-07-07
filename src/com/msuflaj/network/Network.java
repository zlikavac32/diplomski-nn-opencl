package com.msuflaj.network;


import com.msuflaj.transfer.TransferFunction;

public interface Network {

    public double[] process(double[] values)
        throws NetworkException;

    public void process(double[] values, double[] ret)
            throws NetworkException;

    public double[] getWeights();

    public int getWeightsCount();

    public void setWeights(double[] weights);

    public boolean isBiased();

    public TransferFunction[] getTransferFunctions();

    public int[] getDimensions();

}
