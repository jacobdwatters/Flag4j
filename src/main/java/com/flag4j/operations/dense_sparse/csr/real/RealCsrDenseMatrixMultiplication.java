package com.flag4j.operations.dense_sparse.csr.real;


import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
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
     */
    public static Matrix standard(CsrMatrix src1, Matrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(src1.shape, src2.shape);

        double[] destEntries = new double[src1.numRows*src2.numCols];

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;

            for(int aIndex=src1.rowPointers[i]; aIndex<src1.rowPointers[i+1]; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.entries[aIndex];
                int bRowOffset = aCol*src2.numCols;

                for(int bCol = 0; bCol < src2.numCols; bCol++) {
                    double bVal = src2.entries[bRowOffset + bCol];

                    destEntries[rowOffset + bCol] += aVal * bVal;
                }
            }
        }

        return new Matrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }
}
