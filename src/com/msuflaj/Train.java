package com.msuflaj;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPResult;
import com.msuflaj.dataset.CSVLoader;
import com.msuflaj.dataset.DataSet;
import com.msuflaj.dataset.Loader;
import com.msuflaj.encoder.Encoder;
import com.msuflaj.encoder.EquilateralEncoder;
import com.msuflaj.logger.ConsoleLogger;
import com.msuflaj.logger.Logger;
import com.msuflaj.network.FullyConnectedForwardNetwork;
import com.msuflaj.network.Network;
import com.msuflaj.statistics.Statistics;
import com.msuflaj.storage.FileWieghtsStorage;
import com.msuflaj.storage.WeightsStorage;
import com.msuflaj.trainer.*;
import com.msuflaj.transfer.TanHFunction;
import com.msuflaj.transfer.TransferFunction;
import org.jocl.CL;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Properties;
import java.util.Random;

public class Train {

    public static void main(String[] args) throws Exception {

        JSAP parser = new JSAP();

        FlaggedOption opt = new FlaggedOption("config")
                .setStringParser(JSAP.STRING_PARSER)
                .setDefault("conf.properties").setShortFlag('c')
                .setLongFlag("config");

        parser.registerParameter(opt);

        JSAPResult config = parser.parse(args);

        if (!config.success()) {
            printUsage(parser);
        }

        File configFile = new File(config.getString("config"));

        Properties props = new Properties();
        props.load(new BufferedReader(new FileReader(configFile)));

        run(props);

    }

    private static void run(Properties props) throws Exception {

        String dimensions = props.getProperty("network.dimensions");

        if (null == dimensions) {
            error("network.dimensions");
            return ;
        }

        String[] parts = dimensions.split("x|X");

        int[] dimensionsConverted = new int[parts.length];

        if (parts.length < 2) {
            error("There need to be at least input and output layer", false);
            return;
        }

        for (int i = 0; i < parts.length; i++) {
            try {
                dimensionsConverted[i] = Integer.parseInt(parts[i]);
                if (dimensionsConverted[i] < 1) {
                    error("Layer " + (i + 1) + " has neuron count less than 0", false);
                    return;
                }
            } catch (NumberFormatException e) {
                error("network.dimensions");
                return ;
            }
        }

        TransferFunction f = new TanHFunction();

        TransferFunction[] functions = new TransferFunction[dimensionsConverted.length - 1];

        for (int i = 0; i < functions.length; i++) {
            functions[i] = f;
        }

        Network network = new FullyConnectedForwardNetwork(dimensionsConverted, functions, true);

        final int numberOfClasses = dimensionsConverted[dimensionsConverted.length - 1] + 1;

        String algorithm = props.getProperty("train.algorithm").toLowerCase();

        Trainer trainer;

        switch (algorithm) {
            case "rprop":
                trainer = createRPROPTrainer(props);
                break;
            case "backprop":
                trainer = createBackPropTrainer(props);
                break;
            default:
                error("Invalid algorithm", false);
                return;
        }

        String fileName = props.getProperty("train.data");

        if (null == fileName) {
            error("train.data");
            return ;
        }

        File file = new File(fileName);

        Loader loader = new CSVLoader(new FileReader(file), true);
        DataSet dataSet = loader.load(dimensionsConverted[0], new Loader.EncoderFactory() {
            @Override
            public Encoder getEncoder(int n) {
                return new EquilateralEncoder(numberOfClasses);
            }
        });

        int temp;

        try {
            temp = Integer.parseInt(props.getProperty("train.logger.iterationMod"));
        } catch (NumberFormatException e) {
            error("train.logger.iterationMod");
            return ;
        }

        if (temp <= 0) {
            error("Iteration mod must be greater than 0", false);
        }


        Statistics statistics = new Statistics();

        Logger logger = new ConsoleLogger(temp);
        statistics.registerListener(logger, Statistics.ERROR | Statistics.ITERATION | Statistics.FINISH | Statistics.START);

        try {
            temp = Integer.parseInt(props.getProperty("train.iterations"));
        } catch (NumberFormatException e) {
            error("train.iterations");
            return ;
        }

        if (temp < 0) {
            error("Iterations must be greater or equal to 0", false);
        }

        final int numberOfIterations = temp;

        try {
            trainer.train(network, dataSet, new Trainer.StopCondition() {
                @Override
                public boolean isConditionMet(int iteration, double error) {
                    return iteration < numberOfIterations;
                }
            }, statistics);
        } catch (Exception e) {
            error("An error occurred during training", false);
            return ;
        }

        String weightsFile = props.getProperty("network.weights");

        if (null == weightsFile) {
            error("network.weights");
            return;
        }

        File weights = new File(weightsFile);

        WeightsStorage storage = new FileWieghtsStorage(weights);
        storage.store(network);

    }

    private static double[] getWeights(Properties props) {

        String value = props.getProperty("train.weights.low");

        double[] val = new double[2];

        try {
            val[0] = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            error("train.weights.low");
        }

        value = props.getProperty("train.weights.high");

        try {
            val[1] = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            error("train.weights.high");
        }

        if (val[0] >= val[1]) {
            error("Lower weights must have lower value than higher", false);
        }

        return val;
    }

    private static Trainer createBackPropTrainer(Properties props) {
        String value = props.getProperty("train.algorithm.backprop.ni");

        double val;

        try {
            val = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            error("train.algorithm.backprop.ni");
            return null;
        }

        double[] data = getWeights(props);

        if (isOpenCL(props)) {
            return new OpenCLBackPropagationTrainer(val, data[0], data[1], createRandom(props), getOpenCLDevice(props));
        }
        return new BackPropagationTrainer(val, data[0], data[1], createRandom(props));
    }

    private static long getOpenCLDevice(Properties props) {
        String value = props.getProperty("train.implementation.opencl.device");

        long val;

        try {
            val = Long.parseLong(value);
        } catch (NumberFormatException e) {
            error("train.implementation.opencl.device");
            return 0;
        }

        if (CL.CL_DEVICE_TYPE_GPU == val) {
            return CL.CL_DEVICE_TYPE_GPU;
        } else if (CL.CL_DEVICE_TYPE_CPU == val) {
            return CL.CL_DEVICE_TYPE_CPU;
        }

        error("train.implementation.opencl.device");
        return 0;
    }

    private static Random createRandom(Properties props) {
        String value = props.getProperty("train.seed");

        int val;

        try {
            val = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            error("train.seed");
            return null;
        }

        if (val < 0) {
            return new Random(val);
        }

        return new Random();
    }

    private static Trainer createRPROPTrainer(Properties props) {

        int i = 0;
        double[] parts = new double[5];

        for (String k : new String[] {
            "train.algorithm.rprop.kMinus",
            "train.algorithm.rprop.kPlus",
            "train.algorithm.rprop.deltaInitial",
            "train.algorithm.rprop.deltaMin",
            "train.algorithm.rprop.deltaMax"
        }) {
            String value = props.getProperty(k);

            try {
                parts[i++] = Double.parseDouble(value);

                if (parts[i - 1] <= 0) {
                    error(k + " must be greater than 0");
                }

            } catch (NumberFormatException e) {
                error(k);
                return null;
            }
        }

        double[] data = getWeights(props);

        if (parts[0] >= parts[1]) {
            error("kMinus must be lower than kPlus", false);
        }

        if (parts[3] >= parts[4]) {
            error("deltaMin must be lower than deltaMax", false);
        }

        if (isOpenCL(props)) {
            return new OpenCLRPROPTrainer(parts[0], parts[1], parts[2], parts[3], parts[4], data[0], data[1], createRandom(props), getOpenCLDevice(props));
        }
        return new RPROPTrainer(parts[0], parts[1], parts[2], parts[3], parts[4], data[0], data[1], createRandom(props));
    }

    private static boolean isOpenCL(Properties props) {
        String v = props.getProperty("train.implementation");

        if (null == v) {
            error("train.implementation");
            return false;
        }

        v = v.toLowerCase();

        if ("regular".equals(v)) {
            return false;
        } else if ("opencl".equals(v)) {
            return true;
        }

        error("train.implementation");
        return false;

    }

    private static void error(String error, boolean useKey) {
        if (useKey) {
            System.err.println("Invalid value for config key " + error);
        } else {
            System.err.println(error);
        }
        System.exit(2);
    }

    private static void error(String error) {
        error(error, true);
    }

    private static void printUsage(JSAP parser) {
        System.err.println();
        System.err.println("Usage: java -cp:lib/ " + Train.class.getName());
        System.err.println("                " + parser.getUsage());
        System.err.println();
        System.err.println(parser.getHelp());
        System.exit(1);
    }

}
