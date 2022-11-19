package com.flag4j.concurrency.algorithms.subtraction;


import com.flag4j.Matrix;

/**
 * Worker thread for matrix subtraction.
 */
public class MatrixSubtractionWorker extends Thread {

    Matrix src1, src2, dest;
    int rowStart, rowEnd, colStart, colEnd;


    /**
     * Create a worker thread for matrix subtraction.
     * @param dest Destination matrix.
     * @param src1 First matrix in subtraction.
     * @param src2 Second matrix in subtraction.
     * @param rowStart Starting row for worker to subtract.
     * @param rowEnd Ending row for worker to subtract.
     * @param colStart Starting column for worker to subtract.
     * @param colEnd Ending column for worker to subtract.
     */
    public MatrixSubtractionWorker(Matrix dest, Matrix src1, Matrix src2, int rowStart, int rowEnd, int colStart, int colEnd) {
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
                dest.entries[i][j] = src1.entries[i][j] - src2.entries[i][j];
            }
        }
    }
}
