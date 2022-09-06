package com.flag4j;

/**
 * Real Dense Vector.
 */
public class Vector extends Matrix {

    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     *
     * @param entries Entries of the real dense matrix.
     */
    public Vector(double[] entries) {
        super(new double[][]{entries});
    }
}
