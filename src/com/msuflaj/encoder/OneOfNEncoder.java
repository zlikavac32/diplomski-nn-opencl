package com.msuflaj.encoder;

import java.security.InvalidParameterException;

public class OneOfNEncoder implements Encoder {

    private double[][] map;

    public OneOfNEncoder(int n) {
        generate(n);
    }

    private void generate(int n) {

        map = new double[n][n];

        for (int i = 0; i < n; i++) {
            map[i][i] = 1;
        }

    }

    public double[] encode(int i) {
        return map[i];
    }

    public int decode(double[] p) {
        int i = 0;
        for (; i < p.length; i++) {
            if (Math.abs(p[i]) > 1e-9 || p[i] < 0) {
                i++;
                break;
            }
        }
        if (Math.abs(1 - p[i]) > 1e-9 || p[i] < 0) {
            throw new InvalidParameterException("Expected one index with value 1 and all other with value 0");
        }
        int index = i++;
        for (; i < p.length; i++) {
            if (Math.abs(p[i]) > 1e-9 || p[i] < 0) {
                throw new InvalidParameterException("Expected one index with value 1 and all other with value 0");
            }
        }
        return index;
    }

}
