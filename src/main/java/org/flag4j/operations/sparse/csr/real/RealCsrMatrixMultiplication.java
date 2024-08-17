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

package org.flag4j.operations.sparse.csr.real;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.*;

/**
 * This class provides low-level implementation of matrix multiplication between two real CSR matrices.
 */
public final class RealCsrMatrixMultiplication {

    private RealCsrMatrixMultiplication() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static MatrixOld standard(CsrMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        double[] destEntries = new double[src1.numRows*src2.numCols];

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;

            for(int aIndex=src1.rowPointers[i]; aIndex<src1.rowPointers[i+1]; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];

                for(int bIndex=src2.rowPointers[aCol]; bIndex<src2.rowPointers[aCol+1]; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.entries[bIndex];

                    destEntries[rowOffset + bCol] += aVal*bVal;
                }
            }
        }

        return new MatrixOld(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrMatrix, CsrMatrix)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrMatrix standardAsSparse(CsrMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        int[] resultRowPtr = new int[src1.numRows + 1];
        List<Double> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1.numRows; i++) {
            Map<Integer, Double> tempMap = new HashMap<>();
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.entries[bIndex];

                    tempMap.merge(bCol, bVal*aVal, Double::sum);
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

        double[] resultValues = ArrayUtils.fromDoubleList(resultList);
        int[] resultColIndices = ArrayUtils.fromIntegerList(resultColIndexList);

        return new CsrMatrix(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }
}
