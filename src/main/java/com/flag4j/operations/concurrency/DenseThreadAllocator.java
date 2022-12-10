package com.flag4j.operations.concurrency;

import com.flag4j.core.TensorBase;
import com.flag4j.operations.concurrency.algorithms.addition.DenseAdditionWorker;
import com.flag4j.operations.concurrency.algorithms.subtraction.MatrixSubtractionWorker;
import com.flag4j.operations.concurrency.algorithms.transpose.MatrixTransposeWorker;
import com.flag4j.util.ErrorMessages;

import java.util.ArrayList;
import java.util.List;


/**
 * This class contains methods useful for allocating threads to work on tensors concurrently.
 */
public final class DenseThreadAllocator {

    private DenseThreadAllocator() {
        // Hide default constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    public static BinaryThreadFactory<double[]> matrixAddThreadFactory = (double[] dest, double[] src1, double[] src2,
                                                                        int start, int end)
            -> new DenseAdditionWorker(dest, src1, src2, start, end);

    public static BinaryThreadFactory<double[]> matrixSubThreadFactory = (double[] dest, double[] src1, double[] src2,
                                                                        int start, int end)
            -> new MatrixSubtractionWorker(dest, src1, src2, start, end);

    public static UnaryThreadFactory<double[]> matrixTransposeThreadFactory = (double[] dest, double[] src,
                                                                             int start, int end)
            -> new MatrixTransposeWorker(dest, src, start, end);


    /**
     * Allocate and start threads for binary matrix operation.
     * @param dest Destination matrix for transpose.
     * @param src1 Source matrix for transpose.
     * @param src2 Source matrix for transpose.
     * @return A list of {@link DenseAdditionWorker} threads which will concurrently transpose the matrix.
     */
    public static List<Thread> allocateThreads(TensorBase dest, TensorBase src1, TensorBase src2,
                                               BinaryThreadFactory factory) {

        // Evenly distribute work amongst thread size
        List<Thread> threadList = new ArrayList<>();
        int numThreads = Configurations.getNumThreads();
        int chunkSize = Math.max(src1.totalEntries().intValue()/numThreads, 1);
        int end;

        for(int i=0; i<src1.totalEntries().intValue(); i+=chunkSize) {
            end = Math.min(i+chunkSize, src1.totalEntries().intValue());

            threadList.add(factory.makeThread(dest.entries, src1.entries, src2.entries, i, end));
            threadList.get(threadList.size()-1).start(); // Start the thread
        }

        return threadList;
    }


    /**
     * Allocate and start threads for concurrent unary operation.
     * @param dest Destination Tensor for the operation.
     * @param src Source tensor for operation to be applied to.
     * @return A list of worker threads which will concurrently apply operation to the source tensor.
     */
    public static List<Thread> allocateThreads(TensorBase dest, TensorBase src,
                                               UnaryThreadFactory factory) {
        // Evenly distribute work amongst thread size
        List<Thread> threadList = new ArrayList<>();
        int numThreads = Configurations.getNumThreads();
        int chunkSize = src.totalEntries().intValue();
        int end;

        for(int i=0; i<numThreads; i++) {
            end = Math.min(i+chunkSize, src.totalEntries().intValue());

            threadList.add(factory.makeThread(dest.entries, src.entries, i, end));
            threadList.get(threadList.size()-1).start(); // Start the thread
        }

        return threadList;
    }
}
