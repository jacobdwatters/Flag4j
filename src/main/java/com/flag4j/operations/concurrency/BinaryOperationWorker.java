package com.flag4j.operations.concurrency;

/**
 * A worker class for concurrent binary operations.
 * @param <T> Type of array to work on.
 */
public abstract class BinaryOperationWorker<T> extends Thread {
    public T src1, src2, dest;
    public int start, end;


    /**
     * Create a worker thread for binary tensor operations.
     * @param dest Destination Tensor.
     * @param src1 First tensor in operation.
     * @param src2 Second tensor in operation.
     * @param start Starting index for worker to apply operation to.
     * @param end Ending index for worker to apply operation to.
     */
    public BinaryOperationWorker(T dest, T src1, T src2, int start, int end) {
        this.dest = dest;
        this.src1 = src1;
        this.src2 = src2;
        this.start = start;
        this.end = end;
    }
}
