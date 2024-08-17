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

package org.flag4j.operations.dense_sparse.csr.real;


import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.flag4j.core.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains low-level implementations of real sparse-dense matrix multiplication where the sparse matrix
 * is in {@link CsrMatrix CSR} format.
 */
public class RealCsrDenseMatrixMultiplication {

    private RealCsrDenseMatrixMultiplication() {
        // Hide default constructor for utility method.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between a real sparse CSR matrix and a real dense matrix.
     * WARNING: If the first matrix is very large but not very sparse, this method may be slower than converting the
     * first matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link MatrixOld#mult(MatrixOld)}.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static MatrixOld standard(CsrMatrix src1, MatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        double[] destEntries = new double[src1.numRows*src2.numCols];
        int rows1 = src1.numRows;
        int cols2 = src2.numCols;

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*src2.numCols;
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int innerStop = rowOffset + cols2;

            for(int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];
                int src2Idx = aCol*src2.numCols;
                int destIdx = rowOffset;

                while(destIdx < innerStop) {
                    destEntries[destIdx++] += aVal*src2.entries[src2Idx++];
                }
            }
        }

        return new MatrixOld(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between a real dense matrix and a real sparse CSR matrix.
     * WARNING: If the second matrix is very large but not very sparse, this method may be slower than converting the
     * second matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link MatrixOld#mult(MatrixOld)}.
     * @param src1 First matrix in the matrix multiplication (dense matrix).
     * @param src2 Second matrix in the matrix multiplication (sparse CSR matrix).
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static MatrixOld standard(MatrixOld src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        double[] destEntries = new double[src1.numRows * src2.numCols];
        int rows1 = src1.numRows;
        int cols1 = src1.numCols;
        int cols2 = src2.numCols;

        for (int i = 0; i < rows1; i++) {
            int rowOffset = i*cols2;
            int src1RowOffset = i*cols1;

            for (int j = 0; j < cols1; j++) {
                double src1Val = src1.entries[src1RowOffset + j];
                int start = src2.rowPointers[j];
                int stop = src2.rowPointers[j + 1];

                for (int aIndex = start; aIndex < stop; aIndex++) {
                    int aCol = src2.colIndices[aIndex];
                    double aVal = src2.entries[aIndex];
                    destEntries[rowOffset + aCol] += src1Val * aVal;
                }
            }
        }

        return new MatrixOld(new Shape(rows1, cols2), destEntries);
    }


    /**
     * Computes the matrix multiplication between a real sparse CSR matrix and the transpose of a real dense matrix.
     * WARNING: This method is likely slower than {@link #standard(CsrMatrix, MatrixOld) standard(src1, src2.T())} unless
     * {@code src1} has many more columns than rows and is very sparse.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication. Will be implicitly transposed.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same number of rows.
     */
    public static MatrixOld standardTranspose(CsrMatrix src1, MatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.numCols);

        double[] destEntries = new double[src1.numRows*src2.numRows];
        int rows1 = src1.numRows;
        int rows2 = src2.numRows;
        int src2RowOffset;
        int destRowOffset;
        int start;
        int stop;

        for(int k=0; k<rows2; k++) {
            src2RowOffset = k*src2.numCols;

            for(int i=0; i<rows1; i++) {
                destRowOffset = i*src2.numRows + k;
                start = src1.rowPointers[i];
                stop = src1.rowPointers[i+1];

                while(start < stop) {
                    destEntries[destRowOffset] += src1.entries[start]*src2.entries[src2RowOffset + src1.colIndices[start++]];
                }
            }
        }

        return new MatrixOld(new Shape(src1.numRows, src2.numRows), destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a real dense vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 VectorOld in multiplication. Treated as a column vector.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the length of
     * {@code src2}.
     */
    public static VectorOld standardVector(CsrMatrix src1, VectorOld src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.size);

        double[] destEntries = new double[src1.numRows];
        int rows1 = src1.numRows;

        for (int i=0; i<rows1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex = start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];

                destEntries[i] += aVal*src2.entries[aCol];
            }
        }

        return new VectorOld(destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a real sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 VectorOld in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static VectorOld standardVector(CsrMatrix src1, CooVector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.size);

        double[] destEntries = new double[src1.numRows];
        int rows1 = src1.numRows;

        // Iterate over the non-zero elements of the sparse vector.
        for (int k=0; k < src2.entries.length; k++) {
            int col = src2.indices[k];
            double val = src2.entries[k];

            // Perform multiplication only for the non-zero elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        double aVal = src1.entries[aIndex];
                        destEntries[i] += aVal*val;
                    }
                }
            }
        }

        return new VectorOld(destEntries);
    }
}
