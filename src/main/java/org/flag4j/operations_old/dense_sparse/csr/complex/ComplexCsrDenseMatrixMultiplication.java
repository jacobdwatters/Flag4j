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

package org.flag4j.operations_old.dense_sparse.csr.complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrix;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;


/**
 * This class contains low-level implementations of complex-complex sparse-sparse matrix multiplication where the sparse matrices
 * are in CSR format.
 */
public final class ComplexCsrDenseMatrixMultiplication {

    private ComplexCsrDenseMatrixMultiplication() {
        // Hide default constructor for utility method.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between a complex sparse CSR matrix and a complex dense matrix.
     * WARNING: If the first matrix is very large but not very sparse, this method may be slower than converting the
     * first matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link MatrixOld#mult(CMatrixOld)}.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static CMatrixOld standard(CsrCMatrix src1, CMatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src1.numRows*src2.numCols];
        Arrays.fill(destEntries, CNumber.ZERO);
        int rows1 = src1.numRows;
        int cols2 = src2.numCols;

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*src2.numCols;
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int innerStop = rowOffset + cols2;

            for(int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                CNumber aVal = src1.entries[aIndex];
                int src2Idx = aCol*src2.numCols;
                int destIdx = rowOffset;

                while(destIdx < innerStop) {
                    destEntries[destIdx] = destEntries[destIdx++].add(src2.entries[src2Idx++].mult(aVal));
                }
            }
        }

        return new CMatrixOld(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between a complex dense matrix and a complex sparse CSR matrix.
     * WARNING: If the second matrix is very large but not very sparse, this method may be slower than converting the
     * second matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link MatrixOld#mult(MatrixOld)}.
     * @param src1 First matrix in the matrix multiplication (dense matrix).
     * @param src2 Second matrix in the matrix multiplication (sparse CSR matrix).
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static CMatrixOld standard(CMatrixOld src1, CsrCMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src1.numRows * src2.numCols];
        int rows1 = src1.numRows;
        int cols1 = src1.numCols;
        int cols2 = src2.numCols;

        for (int i = 0; i < rows1; i++) {
            int rowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            for (int j = 0; j < cols1; j++) {
                CNumber src1Val = src1.entries[src1RowOffset + j];
                int start = src2.rowPointers[j];
                int stop = src2.rowPointers[j + 1];

                for (int aIndex = start; aIndex < stop; aIndex++) {
                    int aCol = src2.colIndices[aIndex];
                    CNumber aVal = src2.entries[aIndex];
                    destEntries[rowOffset + aCol] = destEntries[rowOffset + aCol].add(src1Val.mult(aVal));
                }
            }
        }

        return new CMatrixOld(new Shape(rows1, cols2), destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex dense vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 VectorOld in multiplication. Treated as a column vector.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the length of
     * {@code src2}.
     */
    public static CVectorOld standardVector(CsrCMatrix src1, CVectorOld src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.size);

        CNumber[] destEntries = new CNumber[src1.numRows];
        Arrays.fill(destEntries, CNumber.ZERO);
        int rows1 = src1.numRows;

        for (int i = 0; i < rows1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex = start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                CNumber aVal = src1.entries[aIndex];

                destEntries[i] = destEntries[i].add(src2.entries[aCol].mult(aVal));
            }
        }

        return new CVectorOld(destEntries);
    }
}
