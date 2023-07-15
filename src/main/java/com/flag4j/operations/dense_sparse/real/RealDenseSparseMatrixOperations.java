package com.flag4j.operations.dense_sparse.real;


import com.flag4j.Matrix;
import com.flag4j.SparseMatrix;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains low-level operations between a real dense and real sparse matrix.
 */
public class RealDenseSparseMatrixOperations {

    private RealDenseSparseMatrixOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix in sum.
     * @param src2 Second matrix in sum.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix add(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] += src2.entries[i];
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix in difference.
     * @param src2 Second matrix in difference.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix sub(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] -= src2.entries[i];
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in the difference.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static Matrix sub(SparseMatrix src2, Matrix src1) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        Matrix dest = new Matrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] *= -1;
            dest.entries[row*src1.numCols + col] += src2.entries[i];
        }

        return dest;
    }


    /**
     * Adds a real dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void addEq(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col] += src2.entries[i];
        }
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in difference.
     * @param src2 Entries of second matrix in difference.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void subEq(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col] -= src2.entries[i];
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a real sparse matrix.
     * @return The result of element-wise multiplication.
     * @param src1 Entries of first matrix in element-wise product.
     * @param src2 Entries of second matrix in element-wise product.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static SparseMatrix elemMult(Matrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        double[] product = new double[src2.nonZeroEntries()];

        for(int i=0; i<product.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            product[i] = src1.entries[row*src1.numCols + col]*src2.entries[i];
        }

        return new SparseMatrix(src2.shape.copy(), product, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a real sparse matrix and a real dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * </p>
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from operations with sparse matrices.
     * </p>
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static SparseMatrix elemDiv(SparseMatrix src1, Matrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        double[] quotient = new double[src1.entries.length];

        int row;
        int col;

        for(int i=0; i<src1.entries.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = src1.entries[i] / src2.entries[row*src2.numCols + col];
        }

        return new SparseMatrix(src1.shape.copy(), quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }
}
