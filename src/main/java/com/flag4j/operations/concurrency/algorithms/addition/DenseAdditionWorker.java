package com.flag4j.operations.concurrency.algorithms.addition;

import com.flag4j.operations.concurrency.BinaryOperationWorker;


/**
 * Worker thread for matrix addition.
 */
public class DenseAdditionWorker extends BinaryOperationWorker<double[]> {

    /**
     * Create a worker thread for matrix addition.
     * @param dest Destination matrix.
     * @param src1 First matrix in addition.
     * @param src2 Second matrix in addition.
     * @param start Starting index for worker to add.
     * @param end Ending index for worker to add.
     */
    public DenseAdditionWorker(double[] dest, double[] src1, double[] src2, int start, int end) {
        super(dest, src1, src2, start, end);
    }


    @Override
    public void run() {
        for(int i=super.start; i<super.end; i++) {
            dest[i] = src1[i] + src2[i];
        }
    }
}
