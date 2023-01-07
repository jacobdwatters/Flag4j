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

package com.flag4j.operations.dense_sparse.real;

import com.flag4j.Shape;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains low level methods for computing the matrix multiplication (and matrix vector multiplication) between
 * a real dense/sparse matrix and a real sparse/dense matrix or vector.
 */
public class RealDenseSparseMatrixMultiplication {

    private RealDenseSparseMatrixMultiplication() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the dense matrix sparse vector multiplication.
     * @param src1 Entries of the dense matrix.
     * @param shape1 Shape of the dense matrix.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in sparse vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] standardVector(double[] src1, Shape shape1, double[] src2, int[] indices) {
        int denseRows = shape1.dims[Axis2D.row()];
        int denseCols = shape1.dims[Axis2D.col()];
        int nonZeros = src2.length;

        double[] dest = new double[denseRows];
        int k;

        for(int i=0; i<denseRows; i++) {
            for(int j=0; j<nonZeros; j++) {
                k = indices[j];
                dest[i] += src1[i*denseCols + k]*src2[j];
            }
        }

        return dest;
    }


    /**
     * Computes the sparse matrix dense vector multiplication.
     * @param src1 Entries of the dense matrix.
     * @param indices Indices of non-zero entries in sparse matrix (row-major).
     * @param shape1 Shape of the sparse matrix.
     * @param src2 Entries of the dense vector.
     * @param shape2 Shape of the dense vector.
     * @return Entries of the dense matrix resulting from the matrix vector multiplication.
     */
    public static double[] standardVector(double[] src1, int[][] indices, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];

        double[] dest = new double[rows1];
        // TODO:

        return dest;
    }
}
