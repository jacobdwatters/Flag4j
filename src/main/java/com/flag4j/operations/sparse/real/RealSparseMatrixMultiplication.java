/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.operations.sparse.real;


import com.flag4j.Shape;
import com.flag4j.operations.concurrency.ThreadManager;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains low level implementations of matrix multiplication for real sparse matrices.
 * <b>WARNING:</b> This class does not provide sanity checks.
 */
public class RealSparseMatrixMultiplication {

    private RealSparseMatrixMultiplication() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero entries for the second sparse matrix.
     * @param colIndices2 column indices of non-zero entries for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static double[] standard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                    double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {

        int rows1 = shape1.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        // r1, c1, r2, and c2 store row/column indices for non-zero values in src1 and src2.
        int r1, c1, r2, c2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = rowIndices2[j]; // = k
                c2 = colIndices2[j]; // = j

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2 + c2] += src1[i]*src2[j];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero entries for the second sparse matrix.
     * @param colIndices2 column indices of non-zero entries for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static double[] concurrentStandard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                    double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {

        int rows1 = shape1.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentLoop(0, src1.length, (i)->{
            int r1 = rowIndices1[i]; // = i
            int c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                int r2 = rowIndices2[j]; // = k
                int c2 = colIndices2[j]; // = j

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2 + c2] += src1[i]*src2[j];
                }
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static double[] standardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                    double[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2] += src1[i]*src2[j];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static double[] concurrentStandardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                          double[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentLoop(0, src1.length, (i) -> {
            int r1 = rowIndices1[i]; // = i
            int c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                int r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2] += src1[i]*src2[j];
                }
            }
        });

        return dest;
    }
}
