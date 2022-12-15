/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.operations;

import com.flag4j.Shape;
import com.flag4j.operations.concurrency.Configurations;
import com.flag4j.operations.concurrency.ThreadManager;
import com.flag4j.util.ErrorMessages;


/**
 * This class contains several algorithms for computing the matrix transpose.
 */
public final class RealDenseTranspose {

    private RealDenseTranspose() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Transposes tensor along specified axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The transpose of the tensor along the specified axes.
     */
    public static double[] standard(double[] src, Shape shape, int axis1, int axis2) {
        // TODO:
        return null;
    }


    /**
     * Transposes a matrix using the standard algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The transpose of the matrix.
     */
    public static double[] standardMatrix(double[] src, final int numRows, final int numCols) {
        double[] transpose = new double[numRows*numCols];

        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                transpose[i + j*numRows] = src[j + i*numCols];
            }
        }

        return transpose;
    }


    /**
     * Transposes a matrix using a blocked algorithm. To get or set the block size see
     * {@link Configurations#getBlockSize()} or {@link Configurations#setBlockSize(int)}.
     * @return The transpose of this tensor along specified axes
     */
    public static double[] blockedMatrix(double[] src, final int numRows, final int numCols) {
        double[] transpose = new double[numRows*numCols];
        final int blockSize = Configurations.getBlockSize();
        int blockRowEnd;
        int blockColEnd;

        for(int i=0; i<numRows; i+=blockSize) {
            for(int j=0; j<numCols; j+=blockSize) {
                blockRowEnd = Math.min(i+blockSize, numRows);
                blockColEnd = Math.min(j+blockSize, numCols);

                // Transpose the block beginning at (i, j)
                for(int blockI=i; blockI<blockRowEnd; blockI++) {
                    for(int blockJ=j; blockJ<blockColEnd; blockJ++) {
                        transpose[blockI + blockJ*numRows] = src[blockJ + blockI*numCols];
                    }
                }

                transpose[i+j*numRows] = src[j+i*numCols];
            }
        }

        return transpose;
    }


    /**
     * Computes the transpose of a matrix using a standard concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The transpose of the source matrix.
     */
    public static double[] standardMatrixConcurrent(double[] src, final int numRows, final int numCols) {
        double[] dest = new double[src.length];

        // Compute transpose concurrently.
        ThreadManager.concurrentLoop(0, numRows, (i) -> {
            for(int j=0; j<numCols; j++) {
                dest[i + j*numRows] = src[j + i*numCols];
            }
        });

        return dest;
    }


    /**
     * Computes the transpose of a matrix using a blocked concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The transpose of the source matrix.
     */
    public static double[] blockedMatrixConcurrent(final double[] src, final int numRows, final int numCols) {
        double[] dest = new double[src.length];
        final int blockSize = Configurations.getBlockSize();

        // Compute transpose concurrently.
        ThreadManager.concurrentLoop(0, numRows, blockSize, (i) -> {
            int blockRowEnd;
            int blockColEnd;

            for(int j=0; j<numCols; j+=blockSize) {
                blockRowEnd = Math.min(i+blockSize, numRows);
                blockColEnd = Math.min(j+blockSize, numCols);

                // Transpose the block beginning at (i, j)
                for(int blockI=i; blockI<Math.min(i+blockSize, numRows); blockI++) {
                    for(int blockJ=j; blockJ<Math.min(j+blockSize, numCols); blockJ++) {
                        dest[blockI + blockJ*numRows] = src[blockJ + blockI*numCols];
                    }
                }

                dest[i+j*numRows] = src[j+i*numCols];
            }
        });

        return dest;
    }
}
