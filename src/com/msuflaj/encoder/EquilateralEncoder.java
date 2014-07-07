package com.msuflaj.encoder;

public class EquilateralEncoder implements Encoder {

    private double[][] map;

    public EquilateralEncoder(int n, double low, double high) {
        generate(n, low, high);
    }

    public EquilateralEncoder(int n) {
        this(n, 0, 1);
    }

    private void generate(int n, double low, double high) {

        map = new double[n][n - 1];

        double diff = high - low;

        final int distance = 1;
        final int distancePow = distance * distance;

        map[1][0] = distance;

        for (int i = 2; i < n; i++) {

            double[] center = map[i];
            int dim = i - 1;

            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < i; k++) {
                    center[j] += map[k][j];
                }
                center[j] = center[j] / i;
            }

            double d = getDistance(center, map[0]);

            center[dim] = Math.sqrt(distancePow - d * d);

        }

        for (int i = 1; i < n; i++) {
            for (int j = n - 2; j >= 0; j--) {
                map[i][j] = map[i][j] / distance * diff + low;
            }
        }

    }

    public double[] encode(int i) {
        return map[i];
    }

    public int decode(double[] p) {
        double min = Double.MAX_VALUE;
        int index = 0;
        for (int i = 0; i < map.length; i++) {
            double d = getDistance(map[i], p);
            if (d < min) {
                min = d;
                index = i;
            }
        }
        return index;
    }

    private double getDistance(double[] f, double[] s) {
        double sum = 0;
        for (int i = 0; i < f.length; i++) {
            double t = s[i] - f[i];
            sum += t * t;
        }
        return Math.sqrt(sum);
    }

}
