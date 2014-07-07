package com.msuflaj.trainer;

import com.msuflaj.dataset.DataSet;
import com.msuflaj.network.FullyConnectedForwardNetwork;
import com.msuflaj.network.Network;
import com.msuflaj.network.NetworkException;
import com.msuflaj.statistics.Statistics;
import com.msuflaj.transfer.TransferFunction;
import com.msuflaj.util.ArrayConverter;
import com.msuflaj.util.ResourceLoader;
import org.jocl.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.jocl.CL.*;

public abstract class OpenCLPropagationTrainer implements Trainer {

    private final double lo;

    private final double hi;

    private final Random random;

    private final long deviceId;

    private double ni;

    public OpenCLPropagationTrainer(double ni, double lo, double hi, Random random) {
        this(ni, lo, hi, random, CL_DEVICE_TYPE_GPU);
    }

    public OpenCLPropagationTrainer(double ni, double lo, double hi, Random random, long deviceId) {
        if (ni < 0) {
            throw new IllegalArgumentException("Ni must be positive");
        }

        if (lo >= hi) {
            throw new IllegalArgumentException("Down boundary must be less than upper boundary");
        }

        this.ni = ni;
        this.lo = lo;
        this.hi = hi;
        this.random = random;
        this.deviceId = deviceId;
    }

    @Override
    public double train(Network net, DataSet dataSet, StopCondition stopCondition, Statistics statistics) throws NetworkException, UnexpectedNetworkException {

        if (!(net instanceof FullyConnectedForwardNetwork)) {
            throw new UnexpectedNetworkException("Expected network must be instance of " + FullyConnectedForwardNetwork.class);
        }

        FullyConnectedForwardNetwork network = (FullyConnectedForwardNetwork) net;

        int[] dimensions = network.getDimensions();
        TransferFunction[] functions = network.getTransferFunctions();
        boolean isBiased = network.isBiased();

        int[] neuronOffsets = new int[dimensions.length];
        neuronOffsets[0] = 0;

        int[] weightsOffsets = new int[dimensions.length - 1];

        int extra = isBiased ? 1 : 0;

        int neuronsOffset = dimensions[0] + extra;
        int weightsOffset = 0;

        for (int i = 1; i < dimensions.length; i++) {

            if (dimensions[i] < 1) {
                throw new NetworkException("Neurons in layer " + i + " must be greater than 0");
            }

            neuronOffsets[i] = neuronsOffset;
            neuronsOffset += dimensions[i] + ((isBiased && (i + 1) < dimensions.length) ? 1 : 0);

            weightsOffsets[i - 1] = weightsOffset;
            weightsOffset += dimensions[i] * (dimensions[i - 1] + extra);

        }

        int dataSetSize = dataSet.first.length;

        float[] outputs = new float[neuronsOffset * dataSetSize];
        float[] rawOutputs = new float[neuronsOffset * dataSetSize];

        if (isBiased) {
            for (int i = 0; i < dataSetSize; i++) {
                int o = neuronsOffset * i;
                for (int j = 0; j < dimensions.length - 1; j++) {
                    outputs[o + neuronOffsets[j] + dimensions[j]] = 1;
                    rawOutputs[o + neuronOffsets[j] + dimensions[j]] = 1;
                }
            }
        }

        int inCount = dimensions[0];
        int outCount = dimensions[dimensions.length - 1];
        int singleDataSetSize = inCount + outCount;

        float[] dataSets = new float[dataSetSize * singleDataSetSize];

        for (int i = 0; i < dataSetSize; i++) {
            int offset = i * singleDataSetSize;
            double[] in = dataSet.first[i];
            double[] out = dataSet.second[i];
            for (int j = 0; j < inCount; j++) {
                dataSets[offset + j] = (float) in[j];
            }
            offset += inCount;
            for (int j = 0; j < outCount; j++) {
                dataSets[offset + j] = (float) out[j];
            }
        }

        float[] weights = new float[weightsOffset];

        double d = hi - lo;

        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (random.nextDouble() * d + lo);
        }

        int[] transferFunctionsIds = new int[functions.length];
        int[] transferFunctionParamsOffsets = new int[functions.length];
        List<Float> params = new ArrayList<>();

        for (int i = 0; i < functions.length; i++) {
            transferFunctionsIds[i] = functions[i].getId();
            transferFunctionParamsOffsets[i] = params.size();
            for (double t : functions[i].getParams()) {
                params.add((float) t);
            }
        }

        float[] transferFunctionParams;

        if (params.size() > 0) {
            transferFunctionParams = new float[params.size()];

            for (int i = 0; i < transferFunctionParams.length; i++) {
                transferFunctionParams[i] = params.get(i);
            }
        } else {
            transferFunctionParams = new float[] { 1 };
        }

        float[] errors = new float[dataSetSize];
        float[] layerErrors = new float[neuronsOffset * dataSetSize];
        float[] gradients = new float[weightsOffset * dataSetSize];
        float[] totalError = new float[] {1};

        Pointer isBiasedPointer = Pointer.to(new int[] {
            isBiased ? 1 : 0
        });
        Pointer numberOfLayersPointer = Pointer.to(new int[] {
            dimensions.length
        });
        Pointer numberOfNeuronsPointer = Pointer.to(new int[] {
            neuronsOffset
        });
        Pointer numberOfWeightsPointer = Pointer.to(new int[] {
                weightsOffset
        });
        Pointer calculateErrorsPointer = Pointer.to(new int[] {
            1
        });
        Pointer niPointer = Pointer.to(new float[] {
            (float) ni
        });
        Pointer dataSetSizePointer = Pointer.to(new int[] {
            dataSetSize
        });
        Pointer datSetsPointer = Pointer.to(dataSets);
        Pointer neuronOffsetsPointer = Pointer.to(neuronOffsets);
        Pointer weightsOffsetsPointer = Pointer.to(weightsOffsets);
        Pointer layerDimensionsPointer = Pointer.to(dimensions);
        Pointer transferFunctionsPointer = Pointer.to(transferFunctionsIds);
        Pointer transferFunctionsParamsPointer = Pointer.to(transferFunctionParams);
        Pointer transferFunctionsParamsOffsetsPointer = Pointer.to(transferFunctionParamsOffsets);
        Pointer weightsPointer = Pointer.to(weights);
        Pointer rawOutputsPointer = Pointer.to(rawOutputs);
        Pointer outputsPointer = Pointer.to(outputs);
        Pointer errorsPointer = Pointer.to(errors);
        Pointer layerErrorsPointer = Pointer.to(layerErrors);
        Pointer gradientsPointer = Pointer.to(gradients);
        Pointer totalErrorPointer = Pointer.to(totalError);

        final int platformIndex = 0;
        final long deviceType = deviceId;
        final int deviceIndex = 0;

        CL.setExceptionsEnabled(true);

        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        int maxWorkGroupSize = (int) getMaxWorkGroupSize(device);

        int lastIteration = dataSetSize % maxWorkGroupSize;
        int workGroupSize = dataSetSize - lastIteration;

        int weightsLastIteration = weightsOffset % maxWorkGroupSize;
        int weightsWorkGroupSize = weightsOffset - weightsLastIteration;

        cl_context context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        // Create a command-queue for the selected device
        cl_command_queue commandQueue =
                clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, null);

        cl_mem neuronOffsetsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * neuronOffsets.length, neuronOffsetsPointer, null
        );

        cl_mem weightsOffsetsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * weightsOffsets.length, weightsOffsetsPointer, null
        );

        cl_mem layerDimensionsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * dimensions.length, layerDimensionsPointer, null
        );

        cl_mem transferFunctionsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * transferFunctionsIds.length, transferFunctionsPointer, null
        );

        cl_mem transferFunctionsParamsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * transferFunctionParams.length, transferFunctionsParamsPointer, null
        );

        cl_mem transferFunctionsParamsOffsetsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * transferFunctionParamsOffsets.length, transferFunctionsParamsOffsetsPointer, null
        );

        cl_mem dataSetsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * dataSets.length, datSetsPointer, null
        );

        cl_mem weightsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * weights.length, weightsPointer, null
        );

        cl_mem rawOutputsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * rawOutputs.length, rawOutputsPointer, null
        );

        cl_mem outputsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * outputs.length, outputsPointer, null
        );

        cl_mem errorsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * errors.length, errorsPointer, null
        );

        cl_mem layerErrorsMemObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * layerErrors.length, layerErrorsPointer, null
        );

        cl_mem gradientsMemoryObject = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * gradients.length, gradientsPointer, null
        );

        cl_mem totalErrorMemoryObject = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float, totalErrorPointer, null
        );

        Pointer neuronOffsetsMemoryObjectPointer = Pointer.to(neuronOffsetsMemoryObject);
        Pointer weightsOffsetsMemoryObjectPointer = Pointer.to(weightsOffsetsMemoryObject);
        Pointer layerDimensionsMemoryObjectPointer = Pointer.to(layerDimensionsMemoryObject);
        Pointer transferFunctionsMemoryObjectPointer = Pointer.to(transferFunctionsMemoryObject);
        Pointer transferFunctionsParamsMemoryObjectPointer = Pointer.to(transferFunctionsParamsMemoryObject);
        Pointer transferFunctionsParamsOffsetsMemoryObjectPointer = Pointer.to(transferFunctionsParamsOffsetsMemoryObject);
        Pointer dataSetsMemoryObjectPointer = Pointer.to(dataSetsMemoryObject);
        Pointer weightsMemoryObjectPointer = Pointer.to(weightsMemoryObject);
        Pointer rawOutputsMemoryObjectPointer = Pointer.to(rawOutputsMemoryObject);
        Pointer outputsMemoryObjectPointer = Pointer.to(outputsMemoryObject);
        Pointer errorsMemoryObjectPointer = Pointer.to(errorsMemoryObject);
        Pointer layerErrorsMemObjectPointer = Pointer.to(layerErrorsMemObject);
        Pointer gradientsMemObjectPointer = Pointer.to(gradientsMemoryObject);
        Pointer totalErrorMemObjectPointer = Pointer.to(totalErrorMemoryObject);

        ResourceLoader loader = new ResourceLoader();
        String programSource = loader.load("/res/cl/trainer.cl");

        // Create the program from the source code
        cl_program program = clCreateProgramWithSource(context,
                2, new String[]{programSource, getUpdateWeightsSource(loader)}, null, null);

        // Build the program
        clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel evaluateKernel = clCreateKernel(program, "evaluate", null);

        clSetKernelArg(evaluateKernel, 0,
                Sizeof.cl_int, isBiasedPointer);

        clSetKernelArg(evaluateKernel, 1,
                Sizeof.cl_int, numberOfLayersPointer);

        clSetKernelArg(evaluateKernel, 2,
                Sizeof.cl_int, numberOfNeuronsPointer);

        clSetKernelArg(evaluateKernel, 3,
                Sizeof.cl_int, calculateErrorsPointer);

        clSetKernelArg(evaluateKernel, 4,
                Sizeof.cl_mem, neuronOffsetsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 5,
                Sizeof.cl_mem, weightsOffsetsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 6,
                Sizeof.cl_mem, layerDimensionsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 7,
                Sizeof.cl_mem, transferFunctionsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 8,
                Sizeof.cl_mem, transferFunctionsParamsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 9,
                Sizeof.cl_mem, transferFunctionsParamsOffsetsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 10,
                Sizeof.cl_mem, dataSetsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 11,
                Sizeof.cl_mem, weightsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 12,
                Sizeof.cl_mem, rawOutputsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 13,
                Sizeof.cl_mem, outputsMemoryObjectPointer);

        clSetKernelArg(evaluateKernel, 14,
                Sizeof.cl_mem, errorsMemoryObjectPointer);



        cl_kernel calculateLayerErrorKernel = clCreateKernel(program, "calculateLayerError", null);

        clSetKernelArg(calculateLayerErrorKernel, 0,
                Sizeof.cl_int, isBiasedPointer);

        clSetKernelArg(calculateLayerErrorKernel, 1,
                Sizeof.cl_int, numberOfLayersPointer);

        clSetKernelArg(calculateLayerErrorKernel, 2,
                Sizeof.cl_int, numberOfNeuronsPointer);

        clSetKernelArg(calculateLayerErrorKernel, 3,
                Sizeof.cl_mem, neuronOffsetsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 4,
                Sizeof.cl_mem, weightsOffsetsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 5,
                Sizeof.cl_mem, layerDimensionsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 6,
                Sizeof.cl_mem, transferFunctionsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 7,
                Sizeof.cl_mem, transferFunctionsParamsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 8,
                Sizeof.cl_mem, transferFunctionsParamsOffsetsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 9,
                Sizeof.cl_mem, dataSetsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 10,
                Sizeof.cl_mem, weightsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 11,
                Sizeof.cl_mem, rawOutputsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 12,
                Sizeof.cl_mem, outputsMemoryObjectPointer);

        clSetKernelArg(calculateLayerErrorKernel, 13,
                Sizeof.cl_mem, layerErrorsMemObjectPointer);


        cl_kernel calculateGradientsKernel = clCreateKernel(program, "calculateGradients", null);

        clSetKernelArg(calculateGradientsKernel, 0,
                Sizeof.cl_int, isBiasedPointer);

        clSetKernelArg(calculateGradientsKernel, 1,
                Sizeof.cl_int, numberOfLayersPointer);

        clSetKernelArg(calculateGradientsKernel, 2,
                Sizeof.cl_int, numberOfNeuronsPointer);

        clSetKernelArg(calculateGradientsKernel, 3,
                Sizeof.cl_int, numberOfWeightsPointer);

        clSetKernelArg(calculateGradientsKernel, 4,
                Sizeof.cl_mem, neuronOffsetsMemoryObjectPointer);

        clSetKernelArg(calculateGradientsKernel, 5,
                Sizeof.cl_mem, weightsOffsetsMemoryObjectPointer);

        clSetKernelArg(calculateGradientsKernel, 6,
                Sizeof.cl_mem, layerDimensionsMemoryObjectPointer);

        clSetKernelArg(calculateGradientsKernel, 7,
                Sizeof.cl_mem, outputsMemoryObjectPointer);

        clSetKernelArg(calculateGradientsKernel, 8,
                Sizeof.cl_mem, layerErrorsMemObjectPointer);

        clSetKernelArg(calculateGradientsKernel, 9,
                Sizeof.cl_mem, gradientsMemObjectPointer);


        cl_kernel sumGradientsKernel = clCreateKernel(program, "sumGradients", null);

        clSetKernelArg(sumGradientsKernel, 0,
                Sizeof.cl_int, numberOfWeightsPointer);

        clSetKernelArg(sumGradientsKernel, 1,
                Sizeof.cl_int, dataSetSizePointer);

        clSetKernelArg(sumGradientsKernel, 2,
                Sizeof.cl_mem, gradientsMemObjectPointer);

        cl_kernel updateWeightsKernel = initUpdateWeightsKernel(context, program, niPointer, gradientsMemObjectPointer, weightsMemoryObjectPointer);

        cl_kernel sumErrorsKernel = clCreateKernel(program, "sumErrors", null);

        clSetKernelArg(sumErrorsKernel, 0,
                Sizeof.cl_int, dataSetSizePointer);

        clSetKernelArg(sumErrorsKernel, 1,
                Sizeof.cl_mem, errorsMemoryObjectPointer);

        clSetKernelArg(sumErrorsKernel, 2,
                Sizeof.cl_mem, totalErrorMemObjectPointer);


        long[] global_work_size = new long[1];

        long[] local_work_size = new long[1];

        long[] global_work_offset = new long[1];


        if (workGroupSize > 0) {

            global_work_offset[0] = 0;
            global_work_size[0] = workGroupSize;
            local_work_size[0] = maxWorkGroupSize;

            clEnqueueNDRangeKernel(commandQueue, evaluateKernel, 1, global_work_offset,
                    global_work_size, local_work_size, 0, null, null);
        }

        if (lastIteration > 0) {

            global_work_offset[0] = workGroupSize;
            global_work_size[0] = lastIteration;
            local_work_size[0] = lastIteration;

            clEnqueueNDRangeKernel(commandQueue, evaluateKernel, 1, global_work_offset,
                    global_work_size, local_work_size, 0, null, null);
        }

        global_work_size[0] = 1;
        local_work_size[0] = 1;

        clEnqueueNDRangeKernel(commandQueue, sumErrorsKernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        clEnqueueReadBuffer(
                commandQueue, totalErrorMemoryObject, CL_TRUE, 0, Sizeof.cl_float, totalErrorPointer, 0, null, null
        );

        double error = totalError[0];
        double bestError = error;

        statistics.signalStart();
        statistics.setError(error);

        for (int i = 0; stopCondition.isConditionMet(i, error); i++) {

            if (workGroupSize > 0) {

                global_work_offset[0] = 0;
                global_work_size[0] = workGroupSize;
                local_work_size[0] = maxWorkGroupSize;

                clEnqueueNDRangeKernel(commandQueue, calculateLayerErrorKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (lastIteration > 0) {

                global_work_offset[0] = workGroupSize;
                global_work_size[0] = lastIteration;
                local_work_size[0] = lastIteration;

                clEnqueueNDRangeKernel(commandQueue, calculateLayerErrorKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (workGroupSize > 0) {

                global_work_offset[0] = 0;
                global_work_size[0] = workGroupSize;
                local_work_size[0] = maxWorkGroupSize;

                clEnqueueNDRangeKernel(commandQueue, calculateGradientsKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (lastIteration > 0) {

                global_work_offset[0] = workGroupSize;
                global_work_size[0] = lastIteration;
                local_work_size[0] = lastIteration;

                clEnqueueNDRangeKernel(commandQueue, calculateGradientsKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (weightsWorkGroupSize > 0) {

                global_work_offset[0] = 0;
                global_work_size[0] = weightsWorkGroupSize;
                local_work_size[0] = maxWorkGroupSize;

                clEnqueueNDRangeKernel(commandQueue, sumGradientsKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (weightsLastIteration > 0) {

                global_work_offset[0] = weightsWorkGroupSize;
                global_work_size[0] = weightsLastIteration;
                local_work_size[0] = weightsLastIteration;

                clEnqueueNDRangeKernel(commandQueue, sumGradientsKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (weightsWorkGroupSize > 0) {

                global_work_offset[0] = 0;
                global_work_size[0] = weightsWorkGroupSize;
                local_work_size[0] = maxWorkGroupSize;

                clEnqueueNDRangeKernel(commandQueue, updateWeightsKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);

            }

            if (weightsLastIteration > 0) {

                global_work_offset[0] = weightsWorkGroupSize;
                global_work_size[0] = weightsLastIteration;
                local_work_size[0] = weightsLastIteration;

                clEnqueueNDRangeKernel(commandQueue, updateWeightsKernel, 1, global_work_offset,
                        global_work_size, local_work_size, 0, null, null);
            }

            if (workGroupSize > 0) {

                global_work_offset[0] = 0;
                global_work_size[0] = workGroupSize;
                local_work_size[0] = maxWorkGroupSize;

                cl_event event = new cl_event();

                clEnqueueNDRangeKernel(commandQueue, evaluateKernel, 1, null,
                        global_work_size, local_work_size, 0, null, event);

                CL.clWaitForEvents(1, new cl_event[] {event});
            }

            if (lastIteration > 0) {

                global_work_offset[0] = workGroupSize;
                global_work_size[0] = lastIteration;
                local_work_size[0] = lastIteration;

                cl_event event = new cl_event();

                clEnqueueNDRangeKernel(commandQueue, evaluateKernel, 1, null,
                        global_work_size, local_work_size, 0, null, event);

                CL.clWaitForEvents(1, new cl_event[] {event});
            }

            statistics.incrementIteration();

        }

        global_work_size[0] = 1;
        local_work_size[0] = 1;

        clEnqueueNDRangeKernel(commandQueue, sumErrorsKernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        clEnqueueReadBuffer(
                commandQueue, totalErrorMemoryObject, CL_TRUE, 0, Sizeof.cl_float, totalErrorPointer, 0, null, null
        );

        error = totalError[0];

        statistics.setError(error);

        if (error < bestError) {
            bestError = error;
        }

        statistics.signalFinish();

        clEnqueueReadBuffer(
                commandQueue, weightsMemoryObject, CL_TRUE,
                0,
                Sizeof.cl_float * weights.length,
                weightsPointer, 0, null, null
        );

        clReleaseMemObject(dataSetsMemoryObject);
        clReleaseMemObject(errorsMemoryObject);
        clReleaseMemObject(gradientsMemoryObject);
        clReleaseMemObject(layerDimensionsMemoryObject);
        clReleaseMemObject(neuronOffsetsMemoryObject);
        clReleaseMemObject(outputsMemoryObject);
        clReleaseMemObject(rawOutputsMemoryObject);
        clReleaseMemObject(totalErrorMemoryObject);
        clReleaseMemObject(transferFunctionsMemoryObject);
        clReleaseMemObject(transferFunctionsParamsMemoryObject);
        clReleaseMemObject(transferFunctionsParamsOffsetsMemoryObject);
        clReleaseMemObject(weightsMemoryObject);
        clReleaseMemObject(weightsOffsetsMemoryObject);

        clReleaseKernel(evaluateKernel);
        clReleaseKernel(sumGradientsKernel);
        clReleaseKernel(calculateGradientsKernel);
        clReleaseKernel(sumErrorsKernel);
        clReleaseKernel(calculateLayerErrorKernel);

        cleanUpKernel();

        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);


        network.setWeights(ArrayConverter.fromFloatToDouble(weights));

        return bestError;
    }

    protected long getMaxWorkGroupSize(cl_device_id device)
    {
        ByteBuffer buffer = ByteBuffer.allocate(
                Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t,
                Pointer.to(buffer), null);
        if (4 == Sizeof.size_t) {
            return buffer.getInt();
        }
        return buffer.getLong();
    }

    protected abstract String getUpdateWeightsSource(ResourceLoader loader);

    protected abstract cl_kernel initUpdateWeightsKernel(cl_context context, cl_program program, Pointer niMemoryObjectPointer, Pointer gradientsMemoryObjectPointer, Pointer weightsMemoryObjectPointer);

    protected abstract void cleanUpKernel();

}
