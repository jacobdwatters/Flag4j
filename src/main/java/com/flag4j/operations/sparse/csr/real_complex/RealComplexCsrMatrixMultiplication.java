package com.flag4j.operations.sparse.csr.real_complex;


import com.flag4j.CMatrix;
import com.flag4j.CsrCMatrix;
import com.flag4j.CsrMatrix;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

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
}