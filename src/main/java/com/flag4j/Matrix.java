package com.flag4j;

import java.util.Arrays;

/**
 * Real Dense Matrix.
 */
public class Matrix {
    /**
     * The values of this matrix.
     */
    double[][] entries;
    protected int m;
    protected int n;


    /**
     * Creates an empty real dense matrix.
     */
    public Matrix() {
        this.m = this.n = 0;
        entries = new double[this.m][this.n];
    }


    /**
     * Constructs a square real dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     */
    public Matrix(int size) {
        this.m = this.n = size;
        this.entries = new double[this.m][this.n];
    }


    /**
     * Creates a square real dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     */
    public Matrix(int size, double value) {
        this.m = this.n = size;
        this.entries = new double[this.m][this.n];
        double[] row = new double[this.n];

        Arrays.fill(row, value);
        Arrays.fill(this.entries, row);
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     */
    public Matrix(int m, int n) {
        this.m = m;
        this.n = n;
        this.entries = new double[this.m][this.n];
    }


    /**
     * Creates a real dense matrix with a specified shape and fills the matrix with the specified value.
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     */
    public Matrix(int m, int n, double value) {
        this.m = m;
        this.n = n;
        this.entries = new double[this.m][this.n];
        double[] row = new double[this.n];

        Arrays.fill(row, value);
        Arrays.fill(this.entries, row);
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(double[][] entries) {
        this.entries = Arrays.stream(entries)
                .map(double[]::clone)
                .toArray(double[][]::new);
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(int[][] entries) {
        this.m = entries.length;
        this.n = entries[0].length;
        this.entries = new double[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = entries[i][j];
            }
        }
    }
}
