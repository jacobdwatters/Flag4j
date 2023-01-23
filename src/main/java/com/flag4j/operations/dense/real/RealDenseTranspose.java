/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.operations.dense.real;

import com.flag4j.Shape;
import com.flag4j.concurrency.Configurations;
import com.flag4j.concurrency.ThreadManager;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;


/**
 * This class contains several low-level algorithms for computing the transpose of real dense tensors.
 */
public final class RealDenseTranspose {

    private RealDenseTranspose() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Transposes tensor along specified axes using a standard transpose algorithm. In this context, transposing a
     * tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The transpose of the tensor along the specified axes.
     */
    public static double[] standard(final double[] src, final Shape shape, final int axis1, final int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        double[] dest = new double[shape.totalEntries().intValue()];
        Shape destShape = shape.copy().swapAxes(axis1, axis2);
        int[] destIndices;

        for(int i=0; i<src.length; i++) {
            destIndices = shape.getIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
            dest[destShape.entriesIndex(destIndices)] = src[i]; // Apply transpose for the element
        }

        return dest;
    }


    /**
     * Transposes tensor along specified axes using a standard concurrent transpose algorithm.
     * In this context, transposing a tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The transpose of the tensor along the specified axes.
     */
    public static double[] standardConcurrent(final double[] src, final Shape shape, final int axis1, final int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        double[] dest = new double[shape.totalEntries().intValue()];
        Shape destShape = shape.copy().swapAxes(axis1, axis2);

        // Compute transpose concurrently
        ThreadManager.concurrentLoop(0, src.length, (i) -> {
            int[] destIndices = shape.getIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
            dest[destShape.entriesIndex(destIndices)] = src[i]; // Apply transpose for the element
        });

        return dest;
    }


    /**
     * Transposes a matrix using the standard algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The transpose of the matrix.
     */
    public static double[] standardMatrix(final double[] src, final int numRows, final int numCols) {
        double[] dest = new double[numRows*numCols];

        int destIndex, srcIndex, end;

        for (int i=0; i<numCols; i++) {
            srcIndex = i;
            destIndex = i*numRows;
            end = destIndex + numRows;

            while (destIndex < end) {
                dest[destIndex++] = src[srcIndex];
                srcIndex += numCols;
            }
        }

        return dest;
    }


    /**
     * Transposes a matrix using a blocked algorithm. To get or set the block size see
     * {@link Configurations#getBlockSize()} or {@link Configurations#setBlockSize(int)}.
     * @param src Source entries to compute transpose of.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The transpose of this tensor along specified axes
     */
    public static double[] blockedMatrix(final double[] src, final int numRows, final int numCols) {
        double[] dest = new double[numRows*numCols];
        final int blockSize = Configurations.getBlockSize();
        int blockRowEnd;
        int blockColEnd;
        int srcIndex, destIndex, end;

        for(int i=0; i<numCols; i+=blockSize) {
            for(int j=0; j<numRows; j+=blockSize) {
                blockRowEnd = Math.min(blockSize, numRows);
                blockColEnd = Math.min(i+blockSize, numCols);

                // Transpose the block beginning at (i, j)
                for(int blockI=i; blockI<blockColEnd; blockI++) {
                    srcIndex = blockI;
                    destIndex = blockI*numRows;
                    end = destIndex + blockRowEnd;

                    while (destIndex < end) {
                        dest[destIndex++] = src[srcIndex];
                        srcIndex += numCols;
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the transpose of a matrix using a standard concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The transpose of the source matrix.
     */
    public static double[] standardMatrixConcurrent(final double[] src, final int numRows, final int numCols) {
        double[] dest = new double[src.length];

        // Compute transpose concurrently.
        ThreadManager.concurrentLoop(0, numCols, (i) -> {
            int srcIndex = i;
            int destIndex = i*numRows;
            int end = destIndex + numRows;

            while (destIndex < end) {
                dest[destIndex++] = src[srcIndex];
                srcIndex += numCols;
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
        ThreadManager.concurrentLoop(0, numCols, blockSize, (i) -> {
            for(int j=0; j<numRows; j+=blockSize) {
                int blockRowEnd = Math.min(blockSize, numRows);
                int blockColEnd = Math.min(i+blockSize, numCols);

                // Transpose the block beginning at (i, j)
                for(int blockI=i; blockI<blockColEnd; blockI++) {
                    int srcIndex = blockI;
                    int destIndex = blockI*numRows;
                    int end = destIndex + blockRowEnd;

                    while (destIndex < end) {
                        dest[destIndex++] = src[srcIndex];
                        srcIndex += numCols;
                    }
                }
            }
        });

        return dest;
    }
}
