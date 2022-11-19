package com.flag4j.concurrency.algorithms.transpose;

import com.flag4j.Matrix;
import com.flag4j.concurrency.Configurations;


/**
 * A worker thread for transposing a specified region of a matrix.
 */
public class MatrixTransposeWorker extends Thread {

    Matrix src, dest;
    int rowStart, rowEnd, colStart, colEnd;


    /**
     * Create a worker thread for matrix transpose.
     * @param dest Destination matrix.
     * @param src Source matrix.
     * @param rowStart Starting row for worker to transpose.
     * @param rowEnd Ending row for worker to transpose.
     * @param colStart Starting column for worker to transpose.
     * @param colEnd Ending column for worker to transpose.
     */
    public MatrixTransposeWorker(Matrix dest, Matrix src, int rowStart, int rowEnd, int colStart, int colEnd) {
        this.dest = dest;
        this.src = src;
        this.rowStart = rowStart;
        this.rowEnd = rowEnd;
        this.colStart = colStart;
        this.colEnd = colEnd;
    }


    @Override
    public void run() {
        int blockSize = Configurations.getBlockSize();

        // Using blocked transpose algorithm
        for (int i=colStart; i<colEnd; i += blockSize) {
            for (int j=rowStart; j<rowEnd; j += blockSize) {
                // transpose the block beginning at [i,j]
                for (int k=i; k<i + blockSize && k<colEnd; k++) {
                    for (int l=j; l<j + blockSize && l<rowEnd; l++) {
                        dest.entries[k][l] = src.entries[l][k];
                    }
                }
            }
        }
    }
}
