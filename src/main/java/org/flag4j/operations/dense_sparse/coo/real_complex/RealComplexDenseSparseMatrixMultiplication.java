/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.operations.dense_sparse.coo.real_complex;


import org.flag4j.complex_numbers.CNumber;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class contains low level methods for computing the matrix multiplication (and matrix vector multiplication) between
 * a real dense/sparse matrix and a real sparse/dense matrix or vector.
 */
public final class RealComplexDenseSparseMatrixMultiplication {

    private RealComplexDenseSparseMatrixMultiplication() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a complex sparse matrix using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] standard(double[] src1, Shape shape1, CNumber[] src2,
                                     int[] rowIndices, int[] colIndices, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        Arrays.fill(dest, CNumber.ZERO);

        int row;
        int col;

        for(int i=0; i<rows1; i++) {
            // Loop over non-zero entries of sparse matrix.
            int destRowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            for(int j=0, len=src2.length; j<len; j++) {
                row = rowIndices[j];
                col = colIndices[j];

                dest[destRowOffset + col] = dest[destRowOffset + col].add(src2[j].mult(src1[src1RowOffset + row]));
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a complex dense matrix using a standard algorithm.
     *
     * @param src1 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] standard(double[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
                                    CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        int row;
        int col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            int destRowOffset = row*cols2;
            int src2RowOffset = col*cols2;

            for(int j=0; j<cols2; j++) {
                dest[destRowOffset + j] = dest[destRowOffset + j].add(src2[src2RowOffset + j].mult(src1[i]));
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a complex sparse matrix using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] concurrentStandard(double[] src1, Shape shape1, CNumber[] src2,
                                              int[] rowIndices, int[] colIndices, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int rowOffset = i*cols2;

                // Loop over non-zero entries of sparse matrix.
                for(int j=0; j<src2.length; j++) {
                    int row = rowIndices[j];
                    int col = colIndices[j];
                    CNumber product = src2[j].mult(src1[i*cols1 + row]);

                    synchronized (dest) {
                        dest[rowOffset + col] = dest[rowOffset + col].add(product);
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a complex dense matrix
     * using a concurrent standard algorithm.
     *
     * @param src1 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] concurrentStandard(double[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
                                              CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(var i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                int rowOffset = row*cols2;

                for(int j=0; j<cols2; j++) {
                    CNumber product = src2[col*cols2 + j].mult(src1[i]);

                    synchronized (dest) {
                        dest[rowOffset + j] = dest[rowOffset + j].add(product);
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a complex sparse matrix using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] standard(CNumber[] src1, Shape shape1, double[] src2,
                                     int[] rowIndices, int[] colIndices, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        int row;
        int col;

        for(int i=0; i<rows1; i++) {
            int destRowOffset = i*cols2;

            // Loop over non-zero entries of sparse matrix.
            for(int j=0; j<src2.length; j++) {
                row = rowIndices[j];
                col = colIndices[j];

                dest[destRowOffset + col] = dest[destRowOffset + col].add(src1[i*cols1 + row].mult(src2[j]));
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a complex dense matrix using a standard algorithm.
     *
     * @param src1 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] standard(CNumber[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
                                     double[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        int row;
        int col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            int destRowOffset = row*cols2;
            int src2RowOffset = col*cols2;

            for(int j=0; j<cols2; j++) {
                dest[destRowOffset + j] = dest[destRowOffset + j].add(src1[i].mult(src2[src2RowOffset + j]));
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a complex sparse matrix using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape2 Shape of the sparse matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] concurrentStandard(CNumber[] src1, Shape shape1, double[] src2,
                                               int[] rowIndices, int[] colIndices, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int destRowOffset = i*cols2;
                int productOffset = i*cols1;

                // Loop over non-zero entries of sparse matrix.
                for(int j=0; j<src2.length; j++) {
                    int row = rowIndices[j];
                    int col = colIndices[j];
                    CNumber product = src1[productOffset + row].mult(src2[j]);

                    synchronized (dest) {
                        dest[destRowOffset + col] = dest[destRowOffset + col].add(product);
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication between a real sparse matrix and a complex dense matrix
     * using a concurrent standard algorithm.
     *
     * @param src1 Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices for non-zero entries of the sparse matrix.
     * @param colIndices Column indices for non-zero entries of the sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense matrix.
     * @param shape2 Shape of the dense matrix.
     * @return The result of the matrix multiplication.
     */
    public static CNumber[] concurrentStandard(CNumber[] src1, int[] rowIndices, int[] colIndices, Shape shape1,
                                               double[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                int rowOffset = row*cols2;

                for(int j=0; j<cols2; j++) {
                    CNumber product = src1[i].mult(src2[col*cols2 + j]);

                    synchronized (dest) {
                        dest[rowOffset + j] = dest[rowOffset + j].add(product);
                    }
                }
            }
        });

        return dest;
    }


    // -------------------- Below are the matrix-vector multiplication algorithms --------------------

    /**
     * Computes the dense matrix sparse vector multiplication using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] standardVector(double[] src1, Shape shape1, CNumber[] src2, int[] indices) {
        int denseRows = shape1.get(0);
        int denseCols = shape1.get(1);
        int nonZeros = src2.length;

        CNumber[] dest = new CNumber[denseRows];
        ArrayUtils.fill(dest, 0);
        int k;

        for(int i=0; i<denseRows; i++) {
            int src1RowOffset = i*denseCols;
            CNumber sum = dest[i];

            for(int j=0; j<nonZeros; j++) {
                k = indices[j];
                sum = sum.add(src2[j].mult(src1[src1RowOffset + k]));
            }

            dest[i] = sum;
        }

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero entries in sparse matrix.
     * @param colIndices Column indices of non-zero entries in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] standardVector(double[] src1, int[] rowIndices, int[] colIndices,
                                          Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);
        int row;
        int col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            dest[row] = dest[row].add(src2[col].mult(src1[i]));
        }

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] blockedVector(double[] src1, Shape shape1, CNumber[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = src2.length;

        int bsize = Configurations.getBlockSize(); // Get the block size to use.

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<rows2; jj += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    int src1RowOffset = i*cols1;
                    CNumber sum = dest[i];

                    for(int j=jj; j<jj+bsize && j<rows2; j++) {
                        sum = sum.add(src2[j].mult(src1[src1RowOffset + indices[j]]));
                    }

                    dest[i] = sum;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] concurrentStandardVector(double[] src1, Shape shape1, CNumber[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = src2.length;

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                CNumber sum = dest[i];

                for(int j=0; j<rows2; j++) {
                    int k = indices[j];
                    sum = sum.add(src2[j].mult(src1[i*cols1 + k]));
                }

                dest[i] = sum;
            }
        });

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero entries in sparse matrix.
     * @param colIndices Column indices of non-zero entries in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] concurrentStandardVector(double[] src1, int[] rowIndices, int[] colIndices,
                                                    Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                CNumber product = src2[col].mult(src1[i]);

                synchronized (dest) {
                    dest[row] = dest[row].add(product);
                }
            }
        });

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] concurrentBlockedVector(double[] src1, Shape shape1, CNumber[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = src2.length;

        final int bsize = Configurations.getBlockSize(); // Get the block size to use.

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        // Blocked matrix-vector multiply
        ThreadManager.concurrentBlockedOperation(rows1, bsize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii += bsize) {
                for(int jj=0; jj<rows2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        CNumber sum = dest[i];

                        for(int j=jj; j<jj+bsize && j<rows2; j++) {
                            int k = indices[j];
                            sum = sum.add(src2[j].mult(src1[i*cols1 + k]));
                        }

                        dest[i] = sum;
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] standardVector(CNumber[] src1, Shape shape1, double[] src2, int[] indices) {
        int denseRows = shape1.get(0);
        int denseCols = shape1.get(1);
        int nonZeros = src2.length;

        CNumber[] dest = new CNumber[denseRows];
        ArrayUtils.fill(dest, 0);
        int k;

        for(int i=0; i<denseRows; i++) {
            CNumber sum = dest[i];

            for(int j=0; j<nonZeros; j++) {
                k = indices[j];
                sum = sum.add(src1[i*denseCols + k].mult(src2[j]));
            }

            dest[i] = sum;
        }

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero entries in sparse matrix.
     * @param colIndices Column indices of non-zero entries in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] standardVector(CNumber[] src1, int[] rowIndices, int[] colIndices,
                                           Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);
        int row;
        int col;

        for(int i=0; i<src1.length; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            dest[row] = dest[row].add(src1[i].mult(src2[col]));
        }

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] blockedVector(CNumber[] src1, Shape shape1, double[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = src2.length;

        int bsize = Configurations.getBlockSize(); // Get the block size to use.

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);
        int k;

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<rows2; jj += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    CNumber sum = dest[i];

                    for(int j=jj; j<jj+bsize && j<rows2; j++) {
                        k = indices[j];
                        sum = sum.add(src1[i*cols1 + k].mult(src2[j]));
                    }

                    dest[i] = sum;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] concurrentStandardVector(CNumber[] src1, Shape shape1, double[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = src2.length;

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                CNumber sum = dest[i];

                for(int j=0; j<rows2; j++) {
                    int k = indices[j];
                    sum = sum.add(src1[i*cols1 + k].mult(src2[j]));
                }

                dest[i] = sum;
            }
        });

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication using a concurrent standard algorithm.
     * @param src1 Entries of the sparse matrix.
     * @param rowIndices Row indices of non-zero entries in sparse matrix.
     * @param colIndices Column indices of non-zero entries in sparse matrix.
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] concurrentStandardVector(CNumber[] src1, int[] rowIndices, int[] colIndices,
                                                     Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int row = rowIndices[i];
                int col = colIndices[i];
                CNumber product = src1[i].mult(src2[col]);

                synchronized (dest) {
                    dest[row] = dest[row].add(product);
                }
            }
        });

        return dest;
    }


    /**
     * Computes the dense matrix sparse vector multiplication using a blocked algorithm.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static CNumber[] concurrentBlockedVector(CNumber[] src1, Shape shape1, double[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = src2.length;

        final int bsize = Configurations.getBlockSize(); // Get the block size to use.

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        // Blocked matrix-vector multiply
        ThreadManager.concurrentBlockedOperation(rows1, bsize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii += bsize) {
                for(int jj=0; jj<rows2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        CNumber sum = dest[i];

                        for(int j=jj; j<jj+bsize && j<rows2; j++) {
                            int k = indices[j];
                            sum = sum.add(src1[i*cols1 + k].mult(src2[j]));
                        }

                        dest[i] = sum;
                    }
                }
            }
        });

        return dest;
    }
}
