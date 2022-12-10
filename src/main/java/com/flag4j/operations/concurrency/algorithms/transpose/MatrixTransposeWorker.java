package com.flag4j.operations.concurrency.algorithms.transpose;

import com.flag4j.operations.concurrency.UnaryOperationWorker;


/**
 * A worker thread for transposing a specified region of a matrix.
 */
public class MatrixTransposeWorker extends UnaryOperationWorker<double[]> {


    /**
     * Create a worker thread for matrix transpose.
     * @param dest Destination matrix.
     * @param src Source matrix.
     * @param start Starting index for worker to transpose.
     * @param end Ending index for worker to transpose.
     */
    public MatrixTransposeWorker(double[] dest, double[] src, int start, int end) {
        super(dest, src, start, end);
    }


    @Override
    public void run() {
        // TODO: Concurrent Implementation
    }
}
