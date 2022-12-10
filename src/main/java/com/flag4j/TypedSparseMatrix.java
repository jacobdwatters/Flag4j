package com.flag4j;

public abstract class TypedSparseMatrix<T> extends TypedMatrix<T> {
    protected int rowIndices, colIndices;

    /**
     * Constructs a typed matrix with a specified size.
     *
     * @param type The type of this matrix.
     * @param m    The number of rows in this matrix.
     * @param n    The number of columns in this matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    protected TypedSparseMatrix(MatrixTypes type, int m, int n) {
        super(type, m, n);
    }
}
