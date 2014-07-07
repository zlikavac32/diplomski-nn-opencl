package com.msuflaj.evaluator;

import com.msuflaj.dataset.DataSet;
import com.msuflaj.network.Network;

public interface Evaluator {

    /**
     *
     * @param network Network to train
     * @param dataSet DataSet to use for evaluation
     * @return Error rate
     */
    public double evaluate(Network network, DataSet dataSet);

}
