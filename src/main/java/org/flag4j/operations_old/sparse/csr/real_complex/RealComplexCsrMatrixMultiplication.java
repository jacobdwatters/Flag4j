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

package org.flag4j.operations_old.sparse.csr.real_complex;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.*;

/**
 * This class provides low-level implementation of matrix multiplication between a real CSR matrix and a complex
 * CSR matrix.
 */
public final class RealComplexCsrMatrixMultiplication {

    private RealComplexCsrMatrixMultiplication() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CMatrixOld standard(CsrMatrixOld src1, CsrCMatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.ensureMatMultShapes(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src1.numRows*src2.numCols];
        Arrays.fill(destEntries, CNumber.ZERO);

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

                    destEntries[rowOffset + bCol] = destEntries[rowOffset + bCol].add(bVal.mult(aVal));
                }
            }
        }

        return new CMatrixOld(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CMatrixOld standard(CsrCMatrixOld src1, CsrMatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.ensureMatMultShapes(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src1.numRows*src2.numCols];
        Arrays.fill(destEntries, CNumber.ZERO);

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

                    destEntries[rowOffset + bCol] = destEntries[rowOffset + bCol].add(aVal.mult(bVal));
                }
            }
        }

        return new CMatrixOld(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrMatrixOld, CsrCMatrixOld)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrCMatrixOld standardAsSparse(CsrMatrixOld src1, CsrCMatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.ensureMatMultShapes(src1.shape, src2.shape);

        int[] resultRowPtr = new int[src1.numRows + 1];
        List<CNumber> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1.numRows; i++) {
            Map<Integer, CNumber> tempMap = new HashMap<>();
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    CNumber bVal = src2.entries[bIndex];

                    tempMap.merge(bCol, bVal.mult(aVal), CNumber::add);
                }
            }

            // Ensure entries within each row are sorted by the column indices.
            List<Integer> tempColIndices = new ArrayList<>(tempMap.keySet());
            Collections.sort(tempColIndices);

            for (int colIndex : tempColIndices) {
                resultColIndexList.add(colIndex);
                resultList.add(tempMap.get(colIndex));
            }

            resultRowPtr[i + 1] = resultList.size();
        }

        CNumber[] resultValues = resultList.toArray(new CNumber[0]);
        int[] resultColIndices = ArrayUtils.fromIntegerList(resultColIndexList);

        return new CsrCMatrixOld(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrCMatrixOld, CsrMatrixOld)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrCMatrixOld standardAsSparse(CsrCMatrixOld src1, CsrMatrixOld src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.ensureMatMultShapes(src1.shape, src2.shape);

        int[] resultRowPtr = new int[src1.numRows + 1];
        List<CNumber> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1.numRows; i++) {
            Map<Integer, CNumber> tempMap = new HashMap<>();
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                CNumber aVal = src1.entries[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.entries[bIndex];

                    tempMap.merge(bCol, aVal.mult(bVal), CNumber::add);
                }
            }

            // Ensure entries within each row are sorted by the column indices.
            List<Integer> tempColIndices = new ArrayList<>(tempMap.keySet());
            Collections.sort(tempColIndices);

            for (int colIndex : tempColIndices) {
                resultColIndexList.add(colIndex);
                resultList.add(tempMap.get(colIndex));
            }

            resultRowPtr[i + 1] = resultList.size();
        }

        CNumber[] resultValues = resultList.toArray(new CNumber[0]);
        int[] resultColIndices = ArrayUtils.fromIntegerList(resultColIndexList);

        return new CsrCMatrixOld(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 VectorOld in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static CVectorOld standardVector(CsrMatrixOld src1, CooCVectorOld src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.ensureEquals(src1.numCols, src2.size);

        CNumber[] destEntries = new CNumber[src1.numRows];
        Arrays.fill(destEntries, CNumber.ZERO);
        int rows1 = src1.numRows;

        // Iterate over the non-zero elements of the sparse vector.
        for (int k=0; k < src2.entries.length; k++) {
            int col = src2.indices[k];
            CNumber val = src2.entries[k];

            // Perform multiplication only for the non-zero elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];
                CNumber destVal = destEntries[col];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        double aVal = src1.entries[aIndex];
                        destVal = destVal.add(val.mult(aVal));
                    }
                }
            }
        }

        return new CVectorOld(destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 VectorOld in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static CVectorOld standardVector(CsrCMatrixOld src1, CooVectorOld src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.ensureEquals(src1.numCols, src2.size);

        CNumber[] destEntries = new CNumber[src1.numRows];
        Arrays.fill(destEntries, CNumber.ZERO);
        int rows1 = src1.numRows;

        // Iterate over the non-zero elements of the sparse vector.
        for (int k=0; k < src2.entries.length; k++) {
            int col = src2.indices[k];
            double val = src2.entries[k];

            // Perform multiplication only for the non-zero elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];
                CNumber destVal = destEntries[i];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        CNumber aVal = src1.entries[aIndex];
                        destVal = destVal.add(aVal.mult(val));
                    }
                }
            }
        }

        return new CVectorOld(destEntries);
    }
}
