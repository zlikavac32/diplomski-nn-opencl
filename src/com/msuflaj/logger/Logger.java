package com.msuflaj.logger;

import com.msuflaj.statistics.Statistics;

import java.text.DecimalFormat;

public abstract class Logger implements Statistics.Listener {

    private static DecimalFormat format = new DecimalFormat("0.00E0");

    private static DecimalFormat formatTime = new DecimalFormat("0.000");

    private int iteration;

    public Logger() {
        this(1);
    }

    public Logger(int iteration) {
        this.iteration = iteration;
    }

    @Override
    public void update(Statistics statistics, int changeType) {

        if ((changeType & Statistics.START) > 0) {
            log("Started");
            return ;
        } else if ((changeType & Statistics.FINISH) > 0) {
            log("Total " + statistics.getIteration() + " iterations in " + formatTime.format(
                (statistics.getFinishTime() - statistics.getStartTime()) * 1e-3
            ) + " seconds");
            log("Best error is " + format.format(statistics.getBestError()));
            return ;
        }

        if ((statistics.getIteration() % iteration) > 0) {
            return ;
        }

        if ((changeType & Statistics.ITERATION) > 0) {
            log("Iteration: " + statistics.getIteration());
        } else if ((changeType & Statistics.ERROR) > 0) {
            log("Error: " + format.format(statistics.getError()));
        } else if ((changeType & Statistics.BEST_ERROR) > 0) {
            log("Best error: " + format.format(statistics.getBestError()));
        }
    }

    public abstract void log(String line);

}
