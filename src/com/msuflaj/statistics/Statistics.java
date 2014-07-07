package com.msuflaj.statistics;

import com.msuflaj.util.Pair;

import java.util.ArrayList;
import java.util.List;

public class Statistics {

    public static final int ERROR = 1;

    public static final int ITERATION = 2;

    public static final int BEST_ERROR = 4;

    public static final int FINISH = 8;

    public static final int START = 16;

    private int iteration;

    private double currentError;

    private double bestError = Double.MAX_VALUE;

    private List<Pair<Listener, Integer>> registeredListeners;

    private long startTime;

    private long finishTime;

    public static interface Listener {

        public void update(Statistics statistics, int changeType);

    }

    public Statistics() {
        registeredListeners = new ArrayList<>();
    }

    public double getError() {
        return currentError;
    }

    public double getBestError() {
        return bestError;
    }

    public int getIteration() {
        return iteration;
    }

    public void incrementIteration() {
        iteration++;
        update(ITERATION);
    }

    public void setError(double error) {
        currentError = error;
        if (error < bestError) {
            bestError = error;
            update(BEST_ERROR);
        }
        update(ERROR);
    }

    public void signalFinish() {
        finishTime = System.currentTimeMillis();
        update(FINISH);
    }

    public void signalStart() {
        startTime = System.currentTimeMillis();
        update(START);
    }

    public long getStartTime() {
        return startTime;
    }

    public long getFinishTime() {
        return finishTime;
    }

    private void update(int what) {
        for (Pair<Listener, Integer> listener : registeredListeners) {
            if ((listener.second & what) > 0) {
                listener.first.update(this, what);
            }
        }
    }

    public void registerListener(Listener listener, int types) {
        registeredListeners.add(new Pair<Listener, Integer>(listener, types));
    }

}
