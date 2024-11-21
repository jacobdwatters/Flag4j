/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.linalg.operations.dense.field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.FieldTensor;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ArrayUtils;

/**
 * Utility class for computing the Hermitian transpose of a dense {@link FieldTensor field tensor}.
 */
public final class DenseFieldHermitianTranspose {


    /**
     * Computes the conjugate transpose of a tensor using a standard implementation. That is, interchanges the axes of the
     * tensor so that it matches the specified axes permutation and so that the axis which are swapped and complex conjugated.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of the tensor along the specified axes.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static <T extends Field<T>> Field<T>[] standardHerm(Field<T>[] src,
                                                               Shape shape,
                                                               int[] axes) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        Field<T>[] dest = new Field[shape.totalEntries().intValue()];
        Shape destShape = shape.permuteAxes(axes);

        for(int i=0, size=src.length; i<size; i++) {
            int[] destIndices = shape.getNdIndices(i);
            ArrayUtils.swapUnsafe(destIndices, axes); // Compute destination indices.
            dest[destShape.getFlatIndex(destIndices)] = src[i].conj(); // Apply conjugate transpose for the element
        }

        return dest;
    }


    /**
     * Computes complex conjugate transpose of a tensor along specified axes using a standard transpose algorithm.
     * In this context, transposing a tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The transpose of the tensor along the specified axes.
     */
    public static <T extends Field<T>> Field<T>[] standardHerm(Field<T>[] src,
                                                               Shape shape,
                                                               int axis1, int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        Field<T>[] dest = new Field[shape.totalEntries().intValue()];
        Shape destShape = shape.swapAxes(axis1, axis2);
        int[] destIndices;

        for(int i=0, size=src.length; i<size; i++) {
            destIndices = shape.getNdIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
            dest[destShape.getFlatIndex(destIndices)] = src[i].conj(); // Apply transpose for the element
        }

        return dest;
    }


    /**
     * Computes complex conjugate transpose of a tensor along specified axes using a standard concurrent transpose algorithm.
     * In this context, transposing a tensor is equivalent to swapping a pair of axes.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axis1 First axis to swap in transpose.
     * @param axis2 Second axis to swap in transpose.
     * @return The complex conjugate transpose of the tensor along the specified axes.
     */
    public static <T extends Field<T>> Field<T>[] standardConcurrentHerm(Field<T>[] src,
                                                                         Shape shape,
                                                                         int axis1, int axis2) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() + " tensor.");
        }

        Field<T>[] dest = new Field[shape.totalEntries().intValue()];
        Shape destShape = shape.swapAxes(axis1, axis2);

        // Compute transpose concurrently
        ThreadManager.concurrentOperation(src.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int[] destIndices = shape.getNdIndices(i);
                ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
                dest[destShape.getFlatIndex(destIndices)] = src[i].conj(); // Apply transpose for the element
            }
        });

        return dest;
    }


    /**
     * Computes the conjugate transpose of a tensor using a concurrent implementation. That is, interchanges the axes of the
     * tensor so that it matches the specified axes permutation and so that the axis which are swapped and complex conjugated.
     * @param src Entries of the tensor.
     * @param shape Shape of the tensor to transpose.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of the tensor along the specified axes.
     * @throws IllegalArgumentException If the {@code axes} array is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If the {@code shape} rank is less than 2.
     */
    public static <T extends Field<T>> Field<T>[] standardConcurrentHerm(Field<T>[] src,
                                                                         Shape shape,
                                                                         int[] axes) {
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new IllegalArgumentException("Tensor transpose not defined for rank " + shape.getRank() +
                    " tensor.");
        }

        Field<T>[] dest = new Field[shape.totalEntries().intValue()];
        Shape destShape = shape.permuteAxes(axes);

        ThreadManager.concurrentOperation(src.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int[] destIndices = shape.getNdIndices(i);
                ArrayUtils.swapUnsafe(destIndices, axes); // Compute destination indices.
                dest[destShape.getFlatIndex(destIndices)] = src[i].conj(); // Apply conjugate transpose for the element
            }
        });

        return dest;
    }



    /**
     * Computes complex conjugate transpose of a matrix using the standard algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The transpose of the matrix.
     */
    public static <T extends Field<T>> Field<T>[] standardMatrixHerm(Field<T>[] src, int numRows, int numCols) {
        Field<T>[] dest = new Field[numRows*numCols];

        int destIndex, srcIndex, end;

        for (int i=0; i<numCols; i++) {
            srcIndex = i;
            destIndex = i*numRows;
            end = destIndex + numRows;

            while(destIndex < end) {
                dest[destIndex++] = src[srcIndex].conj();
                srcIndex += numCols;
            }
        }

        return dest;
    }


    /**
     * Computes complex conjugate transpose of a matrix using a blocked algorithm. To get or set the block size see
     * {@link Configurations#getBlockSize()} or {@link Configurations#setBlockSize(int)}.
     * @param src Source matrix in the transpose.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The complex conjugate transpose of this matrix.
     */
    public static <T extends Field<T>> Field<T>[] blockedMatrixHerm(Field<T>[] src, int numRows, int numCols) {
        Field<T>[] dest = new Field[numRows*numCols];
        int blockSize = Configurations.getBlockSize();
        int blockRowEnd;
        int blockColEnd;
        int srcIndex, destIndex, end;

        for(int i=0; i<numCols; i+=blockSize) {
            for(int j=0; j<numRows; j+=blockSize) {
                blockRowEnd = Math.min(j+blockSize, numRows);
                blockColEnd = Math.min(i+blockSize, numCols);

                // Transpose the block beginning at (i, j)
                for(int blockI=i; blockI<blockColEnd; blockI++) {
                    srcIndex = blockI;
                    destIndex = blockI*numRows;
                    end = destIndex + blockRowEnd;

                    while (destIndex < end) {
                        dest[destIndex++] = src[srcIndex].conj();
                        srcIndex += numCols;
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the complex conjugate transpose of a matrix using a standard concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The complex conjugate transpose of the source matrix.
     */
    public static <T extends Field<T>> Field<T>[] standardMatrixConcurrentHerm(Field<T>[] src,
                                                                               int numRows,
                                                                               int numCols) {
        Field<T>[] dest = new Field[src.length];

        // Compute transpose concurrently.
        ThreadManager.concurrentOperation(numCols, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int srcIndex = i;
                int destIndex = i*numRows;
                int end = destIndex + numRows;

                while (destIndex < end) {
                    dest[destIndex++] = src[srcIndex].conj();
                    srcIndex += numCols;
                }
            }
        });

        return dest;
    }


    /**
     * Computes the complex conjugate transpose of a matrix using a blocked concurrent algorithm.
     * @param src Entries of the matrix to transpose.
     * @param numRows Number of rows in source matrix.
     * @param numCols Number of columns in source matrix.
     * @return The complex conjugate transpose of the source matrix.
     */
    public static <T extends Field<T>> Field<T>[] blockedMatrixConcurrentHerm(Field<T>[] src,
                                                                              int numRows,
                                                                              int numCols) {
        Field<T>[] dest = new Field[src.length];
        int blockSize = Configurations.getBlockSize();

        // Compute transpose concurrently.
        ThreadManager.concurrentBlockedOperation(numCols, blockSize, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                for(int j=0; j<numRows; j+=blockSize) {
                    int blockRowEnd = Math.min(j+blockSize, numRows);
                    int blockColEnd = Math.min(i+blockSize, numCols);

                    // Transpose the block beginning at (i, j)
                    for(int blockI=i; blockI<blockColEnd; blockI++) {
                        int srcIndex = blockI;
                        int destIndex = blockI*numRows;
                        int end = destIndex + blockRowEnd;

                        while (destIndex < end) {
                            dest[destIndex++] = src[srcIndex].conj();
                            srcIndex += numCols;
                        }
                    }
                }
            }
        });

        return dest;
    }
}
