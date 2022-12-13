package com.flag4j.operations.concurrency;

import com.flag4j.Matrix;
import com.flag4j.core.TensorBase;
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


    /**
     * Allocate and start threads for binary matrix operation.
     * @param dest Destination matrix for transpose.
     * @param src1 Source matrix for transpose.
     * @param src2 Source matrix for transpose.
     * @return A list of runnable objects
     */
    public static List<Runnable> allocateThreads(TensorBase dest, TensorBase src1, TensorBase src2,
                                                         BinaryThreadFactory factory) {
        // Evenly distribute work amongst thread size
        int numThreads = Configurations.getNumThreads();
        int chunkSize = Math.max(src1.totalEntries().intValue()/numThreads, 1);
        int end;
        List<Runnable> tasks = new ArrayList<>(numThreads);

        for(int i=0; i<src1.totalEntries().intValue(); i+=chunkSize) {
            end = Math.min(i+chunkSize, src1.totalEntries().intValue());

            tasks.add((factory.makeThread(dest.entries, src1.entries, src2.entries, i, end)));
        }

        return tasks;
    }


    /**
     * Allocate and start threads for concurrent unary operation.
     * @param dest Destination Tensor for the operation.
     * @param src Source tensor for operation to be applied to.
     * @return A list of runnable objects.
     */
    public static List<Runnable> allocateThreads(TensorBase dest, TensorBase src,
                                               UnaryThreadFactory factory) {
        // Evenly distribute work amongst thread size
        int numThreads = Configurations.getNumThreads();
        int chunkSize = Math.max(src.totalEntries().intValue()/numThreads, 1);
        int end;
        List<Runnable> tasks = new ArrayList<>(numThreads);

        for(int i=0; i<src.totalEntries().intValue(); i+=chunkSize) {
            end = Math.min(i+chunkSize, src.totalEntries().intValue());

            tasks.add((factory.makeThread(dest, src, i, end)));
        }

        return tasks;
    }
}
