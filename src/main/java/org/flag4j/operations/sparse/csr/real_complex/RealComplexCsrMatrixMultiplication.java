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

package org.flag4j.operations.sparse.csr.real_complex;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class provides low-level implementation of matrix multiplication between a real CSR matrix and a complex
 * CSR matrix.
 */
public class RealComplexCsrMatrixMultiplication {

    private RealComplexCsrMatrixMultiplication() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CMatrix standard(CsrMatrix src1, CsrCMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src1.numRows*src2.numCols];
        ArrayUtils.fillZeros(destEntries);

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;
            int stop = src1.rowPointers[i+1];

            for(int aIndex=src1.rowPointers[i]; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];
                int innerStop = src2.rowPointers[aCol+1];

                for(int bIndex=src2.rowPointers[aCol]; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    CNumber bVal = src2.entries[bIndex];

                    destEntries[rowOffset + bCol].addEq(bVal.mult(aVal));
                }
            }
        }

        return new CMatrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CMatrix standard(CsrCMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src1.numRows*src2.numCols];
        ArrayUtils.fillZeros(destEntries);

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;
            int stop = src1.rowPointers[i+1];

            for(int aIndex=src1.rowPointers[i]; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                CNumber aVal = src1.entries[aIndex];
                int innerStop = src2.rowPointers[aCol+1];

                for(int bIndex=src2.rowPointers[aCol]; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.entries[bIndex];

                    destEntries[rowOffset + bCol].addEq(aVal.mult(bVal));
                }
            }
        }

        return new CMatrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrMatrix, CsrCMatrix)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrCMatrix standardAsSparse(CsrMatrix src1, CsrCMatrix src2) {
        int[] resultRowPtr = new int[src1.numRows + 1];
        ArrayList<CNumber> resultList = new ArrayList<>();
        ArrayList<Integer> resultColIndexList = new ArrayList<>();

        CNumber[] tempValues = new CNumber[src2.numCols];
        boolean[] hasValue = new boolean[src2.numCols];

        for (int i=0; i<src1.numRows; i++) {
            Arrays.fill(hasValue, false);
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for(int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    CNumber bVal = src2.entries[bIndex];

                    if(!hasValue[bCol]) {
                        tempValues[bCol] = new CNumber(0); // Ensure the value is initialized
                        hasValue[bCol] = true;
                    }
                    tempValues[bCol].addEq(bVal.mult(aVal));
                }
            }

            for(int j=0; j<src2.numCols; j++) {
                if (hasValue[j]) {
                    resultColIndexList.add(j);
                    resultList.add(tempValues[j]);
                }
            }
            resultRowPtr[i + 1] = resultRowPtr[i] + resultColIndexList.size() - (i > 0 ? resultRowPtr[i] : 0);
        }

        CNumber[] resultValues = new CNumber[resultList.size()];
        int[] resultColIndices = new int[resultColIndexList.size()];
        for(int i = 0; i < resultList.size(); i++) {
            resultValues[i] = resultList.get(i);
            resultColIndices[i] = resultColIndexList.get(i);
        }

        return new CsrCMatrix(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static CVector standardVector(CsrMatrix src1, CooCVector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.size);

        CNumber[] destEntries = new CNumber[src1.numRows];
        ArrayUtils.fillZeros(destEntries);
        int rows1 = src1.numRows;

        // Iterate over the non-zero elements of the sparse vector.
        for (int k=0; k < src2.entries.length; k++) {
            int col = src2.indices[k];
            CNumber val = src2.entries[k];

            // Perform multiplication only for the non-zero elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        double aVal = src1.entries[aIndex];
                        destEntries[i].addEq(val.mult(aVal));
                    }
                }
            }
        }

        return new CVector(destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static CVector standardVector(CsrCMatrix src1, CooVector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.size);

        CNumber[] destEntries = new CNumber[src1.numRows];
        ArrayUtils.fillZeros(destEntries);
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
                        CNumber aVal = src1.entries[aIndex];
                        destEntries[i].addEq(aVal.mult(val));
                    }
                }
            }
        }

        return new CVector(destEntries);
    }
}
