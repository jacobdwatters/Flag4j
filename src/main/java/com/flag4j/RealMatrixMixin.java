package com.flag4j;

/**
 * This interface specifies methods which all real matrices should implement.
 * @param <T> Matrix type.
 * @param <W> Complex matrix type.
 */
public interface RealMatrixMixin<T, W> extends
        RealTensorMixin<T, W>,
        MatrixPropertiesMixin<T, Matrix, SparseMatrix, W, T, Double>,
        MatrixOperationsMixin<T, Matrix, SparseMatrix, W, T, Double> {

    /**
     * Checks if a matrix is symmetric. That is, if the matrix is equal to its transpose.
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    boolean isSymmetric();


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    boolean isAntiSymmetric();


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    boolean isOrthogonal();
}
