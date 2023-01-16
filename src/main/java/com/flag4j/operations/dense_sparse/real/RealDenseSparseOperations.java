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

import com.flag4j.Matrix;
import com.flag4j.SparseMatrix;
import com.flag4j.operations.concurrency.util.ErrorMessages;
import com.flag4j.operations.concurrency.util.ParameterChecks;

/**
 * This class contains methods to apply common binary operations to a real dense matrix and to a real sparse matrix.
 */
public class RealDenseSparseOperations {

    private RealDenseSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix add(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] += src2.entries[i];
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix sub(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] -= src2.entries[i];
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix sub(SparseMatrix src2, Matrix src1) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] *= -1;
            dest.entries[row*src1.numCols + col] += src2.entries[i];
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a real sparse matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static SparseMatrix elemMult(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        double[] destEntries = new double[src2.nonZeroEntries()];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src1.entries[row*src1.numCols + col]*src2.entries[i];
        }

        return new SparseMatrix(src2.shape, destEntries, src2.rowIndices, src2.colIndices);
    }
}
