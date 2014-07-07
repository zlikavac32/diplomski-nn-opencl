
package com.msuflaj.trainer;

import com.msuflaj.dataset.DataSet;
import com.msuflaj.network.Network;
import com.msuflaj.network.NetworkException;
import com.msuflaj.statistics.Statistics;
import com.msuflaj.util.ResourceLoader;
import org.jocl.*;

import java.util.Random;

import static org.jocl.CL.*;

public class OpenCLRPROPTrainer extends OpenCLPropagationTrainer {

    private double deltaInitial;

    private double deltaMin;

    private double deltaMax;

    private double kMinus;

    private double kPlus;

    private int weightsSize;

    private cl_kernel updateWeightsKernel;

    private cl_mem previousGradientsMemoryObject;

    private cl_mem deltasMemoryObject;

    public OpenCLRPROPTrainer(double kMinus, double kPlus, double deltaInitial, double deltaMin, double deltaMax, double lo, double hi, Random random, long deviceId) {
        super(0, lo, hi, random, deviceId);
        this.kMinus = kMinus;
        this.kPlus = kPlus;
        this.deltaMin = deltaMin;
        this.deltaMax = deltaMax;
        this.deltaInitial = deltaInitial;
    }

    public OpenCLRPROPTrainer(double kMinus, double kPlus, double deltaInitial, double deltaMin, double deltaMax, double lo, double hi, Random random) {
        this(kMinus, kPlus, deltaInitial, deltaMin, deltaMax, lo, hi, random, CL.CL_DEVICE_TYPE_GPU);
    }

    @Override
    public double train(Network net, DataSet dataSet, StopCondition stopCondition, Statistics statistics) throws NetworkException, UnexpectedNetworkException {
        weightsSize = net.getWeightsCount();
        return super.train(net, dataSet, stopCondition, statistics);
    }

    @Override
    protected String getUpdateWeightsSource(ResourceLoader loader) {
        return loader.load("/res/cl/rprop.cl");
    }

    @Override
    protected cl_kernel initUpdateWeightsKernel(cl_context context, cl_program program, Pointer niMemoryObjectPointer, Pointer gradientsMemoryObjectPointer, Pointer weightsMemoryObjectPointer) {

        updateWeightsKernel = clCreateKernel(program, "updateWeights", null);

        clSetKernelArg(updateWeightsKernel, 0,
                Sizeof.cl_float, Pointer.to(new float[] {
                        (float) kPlus
                }));

        clSetKernelArg(updateWeightsKernel, 1,
                Sizeof.cl_float, Pointer.to(new float[] {
                        (float) kMinus
                }));

        clSetKernelArg(updateWeightsKernel, 2,
                Sizeof.cl_float, Pointer.to(new float[] {
                        (float) deltaMax
                }));

        clSetKernelArg(updateWeightsKernel, 3,
                Sizeof.cl_float, Pointer.to(new float[] {
                        (float) deltaMin
                }));

        clSetKernelArg(updateWeightsKernel, 4,
                Sizeof.cl_mem, gradientsMemoryObjectPointer);

        float[] previousGradients = new float[weightsSize];
        float[] deltas = new float[weightsSize];

        for (int i = 0; i < weightsSize; i++) {
            deltas[i] = (float) deltaInitial;
            previousGradients[i] = 0;
        }

        previousGradientsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * previousGradients.length, Pointer.to(previousGradients), null
        );

        deltasMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * deltas.length, Pointer.to(deltas), null
        );

        clSetKernelArg(updateWeightsKernel, 5,
                Sizeof.cl_mem, Pointer.to(previousGradientsMemoryObject));

        clSetKernelArg(updateWeightsKernel, 6,
                Sizeof.cl_mem, Pointer.to(deltasMemoryObject));

        clSetKernelArg(updateWeightsKernel, 7,
                Sizeof.cl_mem, weightsMemoryObjectPointer);

        return updateWeightsKernel;
    }

    @Override
    protected void cleanUpKernel() {

        if (null == updateWeightsKernel) {
            return ;
        }

        clReleaseMemObject(previousGradientsMemoryObject);
        clReleaseMemObject(deltasMemoryObject);

        clReleaseKernel(updateWeightsKernel);

    }
}
