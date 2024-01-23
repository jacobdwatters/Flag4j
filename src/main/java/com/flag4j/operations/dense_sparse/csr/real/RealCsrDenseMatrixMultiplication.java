package com.flag4j.operations.dense_sparse.csr.real;


import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.Vector;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains low-level implementations of real sparse-dense matrix multiplication where the sparse matrix
 * is in {@link com.flag4j.CsrMatrix CSR} format.
 */
public class RealCsrDenseMatrixMultiplication {

    private RealCsrDenseMatrixMultiplication() {
        // Hide default constructor for utility method.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between a real sparse CSR matrix and a real dense matrix.
     * WARNING: If the first matrix is very large but not very sparse, this method may be slower than converting the
     * first matrix to a {@link CsrMatrix#toDense() dense} matrix and calling {@link Matrix#mult(Matrix)}.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} does not have the same number of columns as {@code src2} has
     * rows.
     */
    public static Matrix standard(CsrMatrix src1, Matrix src2) {
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

        return new Matrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between a real sparse CSR matrix and the transpose of a real dense matrix.
     * WARNING: This method is likely slower than {@link #standard(CsrMatrix, Matrix) standard(src1, src2.T())} unless
     * {@code src1} has many more columns than rows and is very sparse.
     * @param src1 First matrix in the matrix multiplication.
     * @param src2 Second matrix in the matrix multiplication. Will be implicitly transposed.
     * @return The result of the matrix multiplication between {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same number of rows.
     */
    public static Matrix standardTranspose(CsrMatrix src1, Matrix src2) {
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

        return new Matrix(new Shape(src1.numRows, src2.numRows), destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a real dense vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the length of
     * {@code src2}.
     */
    public static Vector standardVector(CsrMatrix src1, Vector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ParameterChecks.assertEquals(src1.numCols, src2.size);

        double[] destEntries = new double[src1.numRows];
        int rows1 = src1.numRows;

        for (int i = 0; i < rows1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex = start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];

                destEntries[i] += aVal*src2.entries[aCol];
            }
        }

        return new Vector(destEntries);
    }
}
