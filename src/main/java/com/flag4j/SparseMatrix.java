package com.flag4j;


/**
 * Sparse Real Matrix.
 */
public class SparseMatrix extends TypedSparseMatrix<double[]> {

    /**
     * Constructs a real sparse matrix with a specified size. The matrix will be filled with zeros.
     *
     * @param m The number of rows in this matrix.
     * @param n The number of columns in this matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    protected SparseMatrix(int m, int n) {
        super(MatrixTypes.SPARSE_MATRIX, m, n);
    }
}
