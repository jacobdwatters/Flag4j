package com.flag4j.operations.sparse.csr.real;

import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;

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
    public static Matrix standard(CsrMatrix src1, CsrMatrix src2) {
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

        return new Matrix(new Shape(src1.numRows, src2.numCols), destEntries);
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
        int[] resultRowPtr = new int[src1.numRows + 1];
        ArrayList<Double> resultList = new ArrayList<>();
        ArrayList<Integer> resultColIndexList = new ArrayList<>();

        double[] tempValues = new double[src2.numCols];
        boolean[] hasValue = new boolean[src2.numCols];

        for (int i=0; i<src1.numRows; i++) {
            Arrays.fill(hasValue, false);

            for (int aIndex=src1.rowPointers[i]; aIndex<src1.rowPointers[i + 1]; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];

                for (int bIndex=src2.rowPointers[aCol]; bIndex<src2.rowPointers[aCol + 1]; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.rowPointers[bIndex];

                    if (!hasValue[bCol]) {
                        tempValues[bCol] = 0; // Ensure the value is initialized
                        hasValue[bCol] = true;
                    }
                    tempValues[bCol] += aVal * bVal;
                }
            }

            for (int j=0; j<src2.numCols; j++) {
                if (hasValue[j]) {
                    resultColIndexList.add(j);
                    resultList.add(tempValues[j]);
                }
            }
            resultRowPtr[i + 1] = resultRowPtr[i] + resultColIndexList.size() - (i > 0 ? resultRowPtr[i] : 0);
        }

        double[] resultValues = new double[resultList.size()];
        int[] resultColIndices = new int[resultColIndexList.size()];
        for (int i = 0; i < resultList.size(); i++) {
            resultValues[i] = resultList.get(i);
            resultColIndices[i] = resultColIndexList.get(i);
        }

        return new CsrMatrix(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }
}
