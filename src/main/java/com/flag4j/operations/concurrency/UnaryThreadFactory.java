package com.flag4j.operations.concurrency;

/**
 * A Thread factory for creating threads for unary, element-wise operations.
 * @param <M> The storage type for the tensor.
 */
public interface UnaryThreadFactory<M> {

    /**
     * Creates a thread for performing a unary tensor operation concurrently.
     * @param dest Destination array for binary operation.
     * @param src Entries of the tensor in the operation.
     * @param start Staring row for thread to work on.
     * @param end Ending row for thread to work on.
     * @return The created thread.
     */
    Runnable makeThread(M dest, M src, int start, int end);
}
