package com.msuflaj.util;

public class ArrayConverter {

    public static void fromDoubleToFloat(double[] data, float[] ret) {
        for (int i = 0; i < data.length; i++) {
            ret[i] = (float) data[i];
        }
    }

    public static float[] fromDoubleToFloat(double[] data) {
        float[] ret = new float[data.length];
        fromDoubleToFloat(data, ret);
        return ret;
    }

    public static void fromFloatToDouble(float[] data, double[] ret) {
        for (int i = 0; i < data.length; i++) {
            ret[i] = data[i];
        }
    }

    public static double[] fromFloatToDouble(float[] data) {
        double[] ret = new double[data.length];
        fromFloatToDouble(data, ret);
        return ret;
    }

}
