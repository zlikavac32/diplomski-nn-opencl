package com.msuflaj.evaluator;

import com.msuflaj.dataset.DataSet;
import com.msuflaj.encoder.Encoder;
import com.msuflaj.network.Network;
import com.msuflaj.network.NetworkException;

public class ClassificationEvaluator implements Evaluator {

    @Override
    public double evaluate(Network network, DataSet dataSet) {
        double error = 0;

        Encoder encoder = dataSet.encoder;

        for (int i = 0; i < dataSet.first.length; i++) {
            try {
                error += Math.abs(encoder.decode(dataSet.second[i]) - encoder.decode(network.process(dataSet.first[i])));
            } catch (NetworkException e) {
                throw new IllegalStateException(e);
            }
        }

        return error / dataSet.first.length;
    }
}
