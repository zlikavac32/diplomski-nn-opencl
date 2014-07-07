package com.msuflaj.dataset;

import au.com.bytecode.opencsv.CSVReader;
import com.msuflaj.encoder.Encoder;

import java.io.IOException;
import java.io.Reader;
import java.util.*;

public class CSVLoader implements Loader {

    private CSVReader reader;

    private boolean skipFirst;

    public CSVLoader(Reader reader) {
        this(reader, false);
    }

    public CSVLoader(CSVReader reader) {
        this(reader, false);
    }

    public CSVLoader(Reader reader, boolean skipFirst) {
        this(new CSVReader(reader), skipFirst);
    }

    public CSVLoader(CSVReader reader, boolean skipFirst) {
        this.reader = reader;
        this.skipFirst = skipFirst;
    }

    @Override
    public DataSet load(int inputCount, int outputCount) throws UnableToLoadException {

        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();
        try {

            if (skipFirst) {
                reader.readNext();
            }

            for (String[] row = reader.readNext(); null != row; row = reader.readNext()) {
                double[] parsedInput = new double[inputCount];
                double[] parsedOutput = new double[outputCount];
                for (int i = 0; i < inputCount; i++) {
                    parsedInput[i] = Double.parseDouble(row[i]);
                }
                for (int i = 0, c = inputCount; i < outputCount; i++, c++) {
                    parsedOutput[i] = Double.parseDouble(row[c]);
                }
                input.add(parsedInput);
                output.add(parsedOutput);
            }

        } catch (IOException e) {
            throw new UnableToLoadException(e);
        }

        return new DataSet(input.toArray(new double[0][0]), output.toArray(new double[0][0]), null);
    }

    @Override
    public DataSet load(int inputCount, EncoderFactory encoderFactory) throws UnableToLoadException {

        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();
        List<String> outputClass = new ArrayList<>();
        SortedSet<String> sortedSet = new TreeSet<>();
        Map<String, Integer> map = new HashMap<>();
        Encoder encoder;
        try {

            if (skipFirst) {
                reader.readNext();
            }

            for (String[] row = reader.readNext(); null != row; row = reader.readNext()) {
                double[] parsedInput = new double[inputCount];
                for (int i = 0; i < inputCount; i++) {
                    parsedInput[i] = Double.parseDouble(row[i]);
                }
                sortedSet.add(row[inputCount]);
                input.add(parsedInput);
                outputClass.add(row[inputCount]);
            }

            encoder = encoderFactory.getEncoder(sortedSet.size());

            int i = 0;

            for (String item : sortedSet) {
                map.put(item, i++);
            }

            for (String item : outputClass) {
                output.add(encoder.encode(map.get(item)));
            }

        } catch (IOException e) {
            throw new UnableToLoadException(e);
        }

        return new DataSet(input.toArray(new double[0][0]), output.toArray(new double[0][0]), encoder);
    }

}
