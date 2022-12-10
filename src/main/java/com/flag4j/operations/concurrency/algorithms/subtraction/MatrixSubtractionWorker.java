package com.flag4j.operations.concurrency.algorithms.subtraction;

import com.flag4j.operations.concurrency.BinaryOperationWorker;

/**
 * Worker thread for matrix subtraction.
 */
public class MatrixSubtractionWorker extends BinaryOperationWorker<double[]> {


    /**
     * Create a worker thread for matrix subtraction.
     * @param dest Destination matrix.
     * @param src1 First matrix in subtraction.
     * @param src2 Second matrix in subtraction.
     * @param start Starting row for worker to subtract.
     * @param end Ending row for worker to subtract.
     */
    public MatrixSubtractionWorker(double[] dest, double[] src1, double[] src2, int start, int end) {
        super(dest, src1, src2, start, end);
    }


    @Override
    public void run() {
        for(int i=super.start; i<super.end; i++) {
            dest[i] = src1[i] - src2[i];
        }
    }
}
