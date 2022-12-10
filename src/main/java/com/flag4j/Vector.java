package com.flag4j;

import com.flag4j.core.VectorBase;
import com.flag4j.core.VectorOrientations;

import java.util.Arrays;


/**
 * Real dense vector. Vectors may be oriented as row vectors, column vectors, or unoriented.
 * See {@link VectorOrientations} for orientations.
 */
public class Vector extends VectorBase<double[]> {


    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public Vector(int size) {
        super(size, VectorOrientations.COL, new double[size]);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public Vector(int size, double fillValue) {
        super(size, VectorOrientations.COL, new double[size]);
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param orientation Orientation of the vector.
     */
    public Vector(int size, VectorOrientations orientation) {
        super(size, orientation, new double[size]);
    }


    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     * @param fillValue Fills array with
     * @param orientation Orientation of the vector.
     */
    public Vector(int size, double fillValue, VectorOrientations orientation) {
        super(size, orientation, new double[size]);
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(double[] entries) {
        super(entries.length, VectorOrientations.COL, entries.clone());
    }

    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public Vector(double[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, entries.clone());
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(int[] entries) {
        super(entries.length, VectorOrientations.COL, new double[entries.length]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }

    /**
     * Creates a vector with specified entries and orientation.
     * @param entries Entries for this column vector.
     * @param orientation Orientation of the vector.
     */
    public Vector(int[] entries, VectorOrientations orientation) {
        super(entries.length, orientation, new double[entries.length] );

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.entries.length, a.getOrientation(), a.entries.clone());
    }
}
