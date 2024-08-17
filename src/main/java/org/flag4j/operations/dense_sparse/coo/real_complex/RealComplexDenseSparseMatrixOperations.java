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

package org.flag4j.operations.dense_sparse.coo.real_complex;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.common.real.RealOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains low level implementations of operations_old between real/complex and dense/sparse matrices.
 */
public final class RealComplexDenseSparseMatrixOperations {

    private RealComplexDenseSparseMatrixOperations() {
        // Hide private constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real dense matrix to a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrixOld add(MatrixOld src1, CooCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrixOld dest = new CMatrixOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] = dest.entries[row*src1.numCols + col].add(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrixOld add(CMatrixOld src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CMatrixOld dest = new CMatrixOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
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
    public static CMatrixOld sub(MatrixOld src1, CooCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CMatrixOld dest = new CMatrixOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].sub(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrixOld sub(CMatrixOld src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrixOld dest = new CMatrixOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].sub(src2.entries[i]);
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
    public static CMatrixOld sub(CooCMatrix src2, MatrixOld src1) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrixOld dest = new CMatrixOld(src1.shape, RealOperations.scalMult(src1.entries, -1));

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
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
    public static CMatrixOld sub(CooMatrix src2, CMatrixOld src1) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrixOld dest = new CMatrixOld(src1.shape, ComplexOperations.scalMult(src1.entries, -1));

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void addEq(CMatrixOld src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.entries[idx] = src1.entries[idx].add(src2.entries[i]);
        }
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void subEq(CMatrixOld src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.entries[idx] = src1.entries[idx].sub(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CooCMatrix elemMult(MatrixOld src1, CooCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src2.entries[i].mult(src1.entries[row*src1.numCols + col]);
        }

        return new CooCMatrix(src2.shape, destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between a complex dense matrix and a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CooCMatrix elemMult(CMatrixOld src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src1.entries[row*src1.numCols + col].mult(src2.entries[i]);
        }

        return new CooCMatrix(src2.shape, destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a complex sparse matrix and a real dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * </p>
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from operations_old with sparse matrices.
     * </p>
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static CooCMatrix elemDiv(CooCMatrix src1, MatrixOld src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] quotient = new CNumber[src1.entries.length];

        int row;
        int col;

        for(int i=0; i<src1.entries.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = src1.entries[i].div(src2.entries[row*src2.numCols + col]);
        }

        return new CooCMatrix(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a real sparse matrix and a complex dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * </p>
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from operations_old with sparse matrices.
     * </p>
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static CooCMatrix elemDiv(CooMatrix src1, CMatrixOld src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] quotient = new CNumber[src1.entries.length];

        int row;
        int col;

        for(int i=0; i<src1.entries.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = new CNumber(src1.entries[i]).div(src2.entries[row*src2.numCols + col]);
        }

        return new CooCMatrix(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col VectorOld to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static CMatrixOld addToEachCol(CooMatrix src, CVectorOld col) {
        CMatrixOld sum = new CMatrixOld(src.numRows, src.numCols);

        for(int j=0; j<sum.numCols; j++) {
            sum.setCol(col, j);
        }

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.entries[idx] = sum.entries[idx].add(src.entries[i]);
        }

        return sum;
    }


    /**
     * Adds a dense vector to add to each row as if the vector is a row vector.
     * @param src Source sparse matrix.
     * @param row VectorOld to add to each row of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each row.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of columns in the {@code src} matrix.
     */
    public static CMatrixOld addToEachRow(CooMatrix src, CVectorOld row) {
        CMatrixOld sum = new CMatrixOld(src.numRows, src.numCols);

        for(int i=0; i<sum.numRows; i++) {
            sum.setRow(row.entries, i);
        }

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.entries[idx] = sum.entries[idx].add(src.entries[i]);
        }

        return sum;
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col VectorOld to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static CMatrixOld addToEachCol(CooCMatrix src, VectorOld col) {
        CMatrixOld sum = new CMatrixOld(src.numRows, src.numCols);

        for(int j=0; j<sum.numCols; j++) {
            sum.setCol(col.entries, j);
        }

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.entries[idx] = sum.entries[idx].add(src.entries[i]);
        }

        return sum;
    }


    /**
     * Adds a dense vector to add to each row as if the vector is a row vector.
     * @param src Source sparse matrix.
     * @param row VectorOld to add to each row of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each row.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of columns in the {@code src} matrix.
     */
    public static CMatrixOld addToEachRow(CooCMatrix src, VectorOld row) {
        CMatrixOld sum = new CMatrixOld(src.numRows, src.numCols);

        for(int i=0; i<sum.numRows; i++) {
            sum.setRow(row.entries, i);
        }

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.entries[idx] = sum.entries[idx].add(src.entries[i]);
        }

        return sum;
    }
}
