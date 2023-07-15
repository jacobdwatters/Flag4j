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

package com.flag4j.operations.dense_sparse.complex;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains methods to apply common binary operations to a complex dense/sparse matrix and to a complex sparse/dense matrix.
 */
public class ComplexDenseSparseOperations {

    private ComplexDenseSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix add(CMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix sub(CMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void addEq(CMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col].addEq(src2.entries[i]);
        }
    }


    /**
     * Subtracts a complex sparse matrix from a complex dense matrix and stores the result in the dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void subEq(CMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }
    }



    /**
     * Computes the element-wise multiplication between a real dense matrix and a real sparse matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static SparseCMatrix elemMult(CMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src1.entries[row*src1.numCols + col].mult(src2.entries[i]);
        }

        return new SparseCMatrix(src2.shape.copy(), destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes element-wise addition of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise subtraction.
     */
    public static CTensor add(CTensor src1, SparseCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensor dest = new CTensor(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[src2.shape.entriesIndex(src2.indices[i])].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes element-wise addition of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise addition.
     */
    public static void addEq(CTensor src1, SparseCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src2.shape.entriesIndex(src2.indices[i])].addEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise tensor a complex sparse tensor from a complex dense tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise tensor subtraction.
     */
    public static CTensor sub(CTensor src1, SparseCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CTensor dest = new CTensor(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[src2.shape.entriesIndex(src2.indices[i])].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes element-wise subtraction of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise subtraction.
     */
    public static void subEq(CTensor src1, SparseCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src2.shape.entriesIndex(src2.indices[i])].subEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise tensor multiplication between a complex dense tensor and a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return THe result of the element-wise tensor multiplication.
     */
    public static SparseCTensor elemMult(CTensor src1, SparseCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];
        int[][] indices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy(src2.indices, indices);

        for(int i=0; i<destEntries.length; i++) {
            destEntries[i] = src1.entries[src2.shape.entriesIndex(src2.indices[i])].mult(src2.entries[i]);
        }

        return new SparseCTensor(src2.shape.copy(), destEntries, indices);
    }


    /**
     * Computes the element-wise division between a complex sparse matrix and a complex dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * </p>
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from operations with sparse matrices.
     * </p>
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static SparseCMatrix elemDiv(SparseCMatrix src1, CMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] quotient = new CNumber[src1.entries.length];

        int row;
        int col;

        for(int i=0; i<src1.entries.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = src1.entries[i].div(src2.entries[row*src2.numCols + col]);
        }

        return new SparseCMatrix(src1.shape.copy(), quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }
}
