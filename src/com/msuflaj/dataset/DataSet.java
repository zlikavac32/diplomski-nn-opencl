package com.msuflaj.dataset;

import com.msuflaj.encoder.Encoder;

public class DataSet {

    public final double[][] first;

    public final double[][] second;

    public final Encoder encoder;

    public DataSet(double[][] first, double[][] second, Encoder encoder) {
        this.first = first;
        this.second = second;
        this.encoder = encoder;
    }

}
