package com.flag4j;

/**
 * Stores the type and shape of matrix object.
 */
abstract class TypedVector<T> {

    /**
     * The type of this matrix.
     */
    public final VectorTypes type;
    /**
     * The values of this matrix.
     */
    T entries;
    /**
     * The number of entries in this vector.
     */
    protected int m;
    /**
     * The orientation of this vector (i.e. Row or column).
     */
    protected VectorOrientations orientation;


    /**
     * Constructs a typed vector with a specified orientation and size.
     * @param type The type of this matrix.
     * @param m The number of rows in this matrix.
     * @param orientation The orientation of the resulting vector.
     */
    protected TypedVector(VectorTypes type, int m, VectorOrientations orientation) {
        this.type = type;
        this.m = m;
        this.orientation = orientation;
    }
}