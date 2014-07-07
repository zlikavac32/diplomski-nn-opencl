package com.msuflaj.trainer;

import com.msuflaj.dataset.DataSet;
import com.msuflaj.network.Network;
import com.msuflaj.network.NetworkException;
import com.msuflaj.statistics.Statistics;

public interface Trainer {

    public static interface StopCondition {

        public boolean isConditionMet(int iteration, double error);

    }

    /**
     * Trains network
     * @param network Network to train
     * @param dataSet DataSet that is used to train
     * @param stopCondition Determines if we should stop training
     * @return Error rate
     */
    public double train(Network network, DataSet dataSet, StopCondition stopCondition, Statistics statistics) throws NetworkException, UnexpectedNetworkException;

}
