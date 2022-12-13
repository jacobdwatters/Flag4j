package com.flag4j.operations.concurrency;

/**
 * A factory for creating threads for binary, element-wise operations.
 * @param <M> The storage type for the tensor.
 */
public interface BinaryThreadFactory<M> {

    /**
     * Creates a thread for performing a binary tensor operation concurrently.
     * @param dest Destination array for binary operation.
     * @param src1 Entries of first tensor in the operation.
     * @param src2 Entries of second tensor in the operation.
     * @param start Staring index for thread to work on.
     * @param end Ending index thread to work on.
     * @return The created thread.
     */
    Runnable makeThread(M dest, M src1, M src2, int start, int end);
}
