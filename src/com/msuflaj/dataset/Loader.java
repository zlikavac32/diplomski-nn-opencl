package com.msuflaj.dataset;

import com.msuflaj.encoder.Encoder;

public interface Loader {

    public static interface EncoderFactory {

        public Encoder getEncoder(int n);

    }

    public DataSet load(int inputCount, int outputCount) throws UnableToLoadException;

    public DataSet load(int inputCount, EncoderFactory encoderFactory) throws UnableToLoadException;

}
