package com.msuflaj.logger;

public class ConsoleLogger extends Logger {

    public ConsoleLogger() {
        this(1);
    }

    public ConsoleLogger(int iteration) {
        super(iteration);
    }

    @Override
    public void log(String line) {
        System.out.println(line);
    }
}
