package com.flag4j;

import com.flag4j.util.ErrorMessages;

/**
 * Stores the type and shape of matrix object.
 * @param <T> The type of the entry of this matrix.
 */
public abstract class TypedMatrix<T> {

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
     * @throws IllegalArgumentException if either m or n is negative.
     */
    protected TypedMatrix(MatrixTypes type, int m, int n) {
        if(m<0 || n<0) {
            throw new IllegalArgumentException(
                    ErrorMessages.negativeDimErrMsg(this.getShape())
            );
        }

        this.m = m;
        this.n = n;
        this.type = type;
    }


    /**
     * Gets the shape of this tensor.
     * @return A shape object describing the
     */
    public Shape getShape() {
        return new Shape(m, n);
    }
}
