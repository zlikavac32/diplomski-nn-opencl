package com.msuflaj.storage;

import com.msuflaj.network.Network;

import java.io.*;
import java.nio.ByteBuffer;

public class FileWieghtsStorage implements WeightsStorage {

    private final File file;

    public FileWieghtsStorage(File file) {
        this.file = file;
    }

    @Override
    public boolean store(Network net) {
        BufferedOutputStream stream = null;
        try {
            double[] weights = net.getWeights();
            byte[] bytes = new byte[weights.length * Double.SIZE >> 3];
            ByteBuffer buffer = ByteBuffer.wrap(bytes);
            for (double d : weights) {
                buffer.putDouble(d);
            }
            stream = new BufferedOutputStream(new FileOutputStream(file));
            stream.write(bytes);
        } catch (IOException e) {
            return false;
        } finally {
            if (null != stream) {
                try {
                    stream.close();
                } catch (IOException ignored) {

                }
            }
        }
        return true;
    }

    @Override
    public boolean load(Network net) {
        BufferedInputStream stream = null;
        try {
            double[] weights = new double[net.getWeightsCount()];
            byte[] bytes = new byte[weights.length * Double.SIZE >> 3];
            stream = new BufferedInputStream(new FileInputStream(file));
            if (weights.length * (Double.SIZE >> 3) != stream.read(bytes)) {
                stream.close();
                return false;
            }
            ByteBuffer buffer = ByteBuffer.wrap(bytes);
            for (int i = 0; i < weights.length; i++) {
                weights[i] = buffer.getDouble();
            }
            net.setWeights(weights);
        } catch (IOException e) {
            return false;
        } finally {
            if (null != stream) {
                try {
                    stream.close();
                } catch (IOException ignored) {

                }
            }
        }
        return true;
    }
}
