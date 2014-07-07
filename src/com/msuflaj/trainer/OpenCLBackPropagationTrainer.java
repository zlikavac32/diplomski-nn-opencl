package com.msuflaj.trainer;

import com.msuflaj.util.ResourceLoader;
import org.jocl.*;

import java.util.Random;

import static org.jocl.CL.*;

public class OpenCLBackPropagationTrainer extends OpenCLPropagationTrainer {


    private cl_kernel updateWeightsKernel;

    public OpenCLBackPropagationTrainer(double ni, double lo, double hi, Random random, long deviceId) {
        super(ni, lo, hi, random);
    }

    @Override
    protected String getUpdateWeightsSource(ResourceLoader loader) {
        return loader.load("/res/cl/back_propagation.cl");
    }

    public OpenCLBackPropagationTrainer(double ni, double lo, double hi, Random random) {
        this(ni, lo, hi, random, CL_DEVICE_TYPE_GPU);
    }


    @Override
    protected cl_kernel initUpdateWeightsKernel(cl_context context, cl_program program, Pointer niMemoryObjectPointer, Pointer gradientsMemoryObjectPointer, Pointer weightsMemoryObjectPointer) {
        updateWeightsKernel = clCreateKernel(program, "updateWeights", null);

        clSetKernelArg(updateWeightsKernel, 0,
                Sizeof.cl_float, niMemoryObjectPointer);

        clSetKernelArg(updateWeightsKernel, 1,
                Sizeof.cl_mem, gradientsMemoryObjectPointer);

        clSetKernelArg(updateWeightsKernel, 2,
                Sizeof.cl_mem, weightsMemoryObjectPointer);

        return updateWeightsKernel;
    }

    @Override
    protected void cleanUpKernel() {
        if (null == updateWeightsKernel) {
            return ;
        }
        clReleaseKernel(updateWeightsKernel);
    }

}
