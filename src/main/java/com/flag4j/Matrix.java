package com.flag4j;

import java.util.Arrays;

/**
 * Real Dense Matrix.
 */
public class Matrix {
    double[][] entries;
    int m, n;

    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(double[][] entries) {
        this.entries = Arrays.stream(entries)
                .map(double[]::clone)
                .toArray(double[][]::new);
        this.m = this.entries.length;
        this.n = this.entries[0].length;
    }
}
