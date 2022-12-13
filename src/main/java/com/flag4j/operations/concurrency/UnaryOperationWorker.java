package com.flag4j.operations.concurrency;

/**
 * Worker for unary operations.
 * @param <T> Storage type for the tensor.
 */
public abstract class UnaryOperationWorker<T> implements Runnable {
    T src, dest;
    int start, end;


    /**
     * Create a worker thread for unary tensor operations.
     * @param dest Destination Tensor.
     * @param src Tensor in operation.
     * @param start Starting index for worker to apply operation to.
     * @param end Ending index for worker to apply operation to.
     */
    public UnaryOperationWorker(T dest, T src, int start, int end) {
        this.dest = dest;
        this.src = src;
        this.start = start;
        this.end = end;
    }
}
