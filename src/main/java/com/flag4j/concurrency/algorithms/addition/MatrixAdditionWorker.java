package com.flag4j.concurrency.algorithms.addition;

import com.flag4j.Matrix;


/**
 * Worker thread for matrix addition.
 */
public class MatrixAdditionWorker extends Thread {

    Matrix src1, src2, dest;
    int rowStart, rowEnd, colStart, colEnd;


    /**
     * Create a worker thread for matrix addition.
     * @param dest Destination matrix.
     * @param src1 First matrix in addition.
     * @param src2 Second matrix in addition.
     * @param rowStart Starting row for worker to add.
     * @param rowEnd Ending row for worker to add.
     * @param colStart Starting column for worker to add.
     * @param colEnd Ending column for worker to add.
     */
    public MatrixAdditionWorker(Matrix dest, Matrix src1, Matrix src2, int rowStart, int rowEnd, int colStart, int colEnd) {
        this.dest = dest;
        this.src1 = src1;
        this.src2 = src2;
        this.rowStart = rowStart;
        this.rowEnd = rowEnd;
        this.colStart = colStart;
        this.colEnd = colEnd;
    }


    @Override
    public void run() {
        for(int i=rowStart; i<rowEnd; i++) {
            for(int j=colStart; j<colEnd; j++) {
                dest.entries[i][j] = src1.entries[i][j] + src2.entries[i][j];
            }
        }
    }
}
