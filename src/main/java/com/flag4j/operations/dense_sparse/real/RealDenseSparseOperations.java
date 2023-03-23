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

import com.flag4j.*;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.Arrays;

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
     * @param src1 First matrix in sum.
     * @param src2 Second matrix in sum.
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
     * Adds a real dense tensor to a real sparse tensor.
     * @param src1 First tensor in sum.
     * @param src2 Second tensor in sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static Tensor add(Tensor src1, SparseTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int[] indices;
        Tensor dest = new Tensor(src1);

        for(int i=0; i<src2.entries.length; i++) {
            indices = src2.indices[i];
            dest.entries[dest.shape.entriesIndex(indices)] += src2.entries[i];
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix in difference.
     * @param src2 Second matrix in difference.
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
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in the difference.
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
     * Adds a real dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void addEq(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col] += src2.entries[i];
        }
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in difference.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void subEq(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col] -= src2.entries[i];
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a real sparse matrix.
     * @return The result of element-wise multiplication.
     * @param src1 Entries of first matrix in element-wise product.
     * @param src2 Entries of second matrix in element-wise product.
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

        return new SparseMatrix(src2.shape.copy(), destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a real sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result ofm element-wise multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static SparseTensor elemMult(Tensor src1, SparseTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        double[] destEntries = new double[src2.nonZeroEntries()];
        int[][] destIndices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy(src2.indices, destIndices);

        for(int i=0; i<destEntries.length; i++) {
            index = src2.shape.entriesIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.entries[index]*src2.entries[i];
        }

        return new SparseTensor(src2.shape.copy(), destEntries, destIndices);
    }


    /**
     * Subtracts a real sparse tensor from a real dense tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static Tensor sub(Tensor src1, SparseTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        Tensor dest = new Tensor(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[dest.shape.entriesIndex(src2.indices[i])] -= src2.entries[i];
        }

        return dest;
    }


    /**
     * Adds a real dense tensor to a real sparse tensor and stores the result in the first tensor.
     * @param src1 First tensor in sum. Also, storage of result.
     * @param src2 Second tensor in sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static void addEq(Tensor src1, SparseTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src1.shape.entriesIndex(src2.indices[i])] += src2.entries[i];
        }
    }


    /**
     * Subtracts a real sparse tensor from a real dense tensor and stores the result in the dense tensor.
     * @param src1 First tensor in difference. Also, storage of result.
     * @param src2 Second tensor in difference.
     * @return The result of the tensor subtraction.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static void subEq(Tensor src1, SparseTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src1.shape.entriesIndex(src2.indices[i])] -= src2.entries[i];
        }
    }
}
