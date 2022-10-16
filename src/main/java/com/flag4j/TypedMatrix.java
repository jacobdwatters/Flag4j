package com.flag4j;

/**
 * Stores the type and shape of matrix object.
 */
abstract class TypedMatrix<T> {

    /**
     * The type of this matrix.
     */
    public final MatrixTypes type;
    /**
     * The values of this matrix.
     */
    T entries;
    /**
     * The number of rows in this matrix.
     */
    protected int m;
    /**
     * The number of columns in this matrix.
     */
    protected int n;


    /**
     * Constructs a typed matrix with a specified size.
     * @param type The type of this matrix.
     * @param m The number of rows in this matrix.
     * @param n The number of columns in this matrix.
     */
    protected TypedMatrix(MatrixTypes type, int m, int n) {
        this.m = m;
        this.n = n;
        this.type = type;
    }
}
