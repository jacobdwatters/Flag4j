package com.flag4j;

import java.util.Arrays;

/**
 * Real Dense Vector.
 */
public class Vector extends TypedVector<double[]> {

    /**
     * Creates an empty column vector of size 0
     */
    public Vector() {
        super(VectorTypes.VECTOR, 0, VectorOrientations.COLUMN_VECTOR);
        entries = new double[this.m];
    }


    /**
     * Creates an empty vector of size 0 with a specified orientation.
     * @param orientation The orientation of this vector.
     */
    public Vector(VectorOrientations orientation) {
        super(VectorTypes.VECTOR, 0, orientation);
        entries = new double[this.m];
    }


    /**
     * Creates a column vector filled with zeros.
     * @param size The number of entries in this column vector.
     */
    public Vector(int size) {
        super(VectorTypes.VECTOR, size, VectorOrientations.COLUMN_VECTOR);
        entries = new double[this.m];
    }


    /**
     * Creates a vector filled with zeros with a specified orientation.
     * @param size The number of entries in this column vector.
     * @param orientation The orientation of this vector.
     */
    public Vector(int size, VectorOrientations orientation) {
        super(VectorTypes.VECTOR, size, orientation);
        entries = new double[this.m];
    }


    /**
     * Creates a column vector with a specified size and fill value.
     * @param size The number of entries in this column vector.
     * @param value The fill value for this vector.
     */
    public Vector(int size, double value) {
        super(VectorTypes.VECTOR, size, VectorOrientations.COLUMN_VECTOR);
        entries = new double[this.m];
        Arrays.fill(entries, value);
    }


    /**
     * Creates a vector with a specified size, fill value, and orientation.
     * @param size The number of entries in this column vector.
     * @param value The fill value for this vector.
     * @param orientation The orientation of this vector.
     */
    public Vector(int size, double value, VectorOrientations orientation) {
        super(VectorTypes.VECTOR, size, orientation);
        entries = new double[this.m];
        Arrays.fill(entries, value);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries The entries for this column vector.
     */
    public Vector(double... entries) {
        super(VectorTypes.VECTOR, entries.length, VectorOrientations.COLUMN_VECTOR);
        this.entries = entries.clone();
    }


    /**
     * Creates a vector with specified entries and orientation.
     * @param entries The entries for this column vector.
     * @param orientation The orientation of this vector.
     */
    public Vector(double[] entries, VectorOrientations orientation) {
        super(VectorTypes.VECTOR, entries.length, orientation);
        this.entries = entries.clone();
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries The entries for this column vector.
     */
    public Vector(int... entries) {
        super(VectorTypes.VECTOR, entries.length, VectorOrientations.COLUMN_VECTOR);
        this.entries = Arrays.stream(entries).asDoubleStream().toArray();
    }


    /**
     * Creates a vector with specified entries and orientation.
     * @param entries The entries for this column vector.
     * @param orientation The orientation of this vector.
     */
    public Vector(int[] entries, VectorOrientations orientation) {
        super(VectorTypes.VECTOR, entries.length, orientation);
        this.entries = Arrays.stream(entries).asDoubleStream().toArray();
    }


    /**
     * Creates a vector with orientation and entries specified by another vector.
     * @param a The vector which specifies the orientation and entries of this vector.
     */
    public Vector(Vector a) {
        super(VectorTypes.VECTOR, a.entries.length, a.orientation);
        this.entries = a.entries.clone();
    }


    /**
     * Creates a vector with entries specified by another vector. The orientation of the resulting vector is specified separately and need not
     * match the orientation of the vector a.
     * @param a The vector which specifies the entries of this vector.
     * @param orientation The orientation of the vector.
     */
    public Vector(Vector a, VectorOrientations orientation) {
        super(VectorTypes.VECTOR, a.entries.length, orientation);
        this.entries = a.entries.clone();
    }
}
