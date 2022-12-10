package com.flag4j;

import com.flag4j.core.MatrixBase;
import com.flag4j.operations.RealDenseOperations;
import com.flag4j.util.ShapeChecks;

import java.util.Arrays;

/**
 * Real dense matrix. Stored in row major format.
 */
public class Matrix extends MatrixBase<double[]> {


    /**
     * Constructs a square real dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size) {
        super(new Shape(size, size), new double[size*size]);
    }


    /**
     * Creates a square real dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size, double value) {
        super(new Shape(size, size), new double[size*size]);
        Arrays.fill(super.entries, value);
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols) {
        super(new Shape(rows, cols), new double[rows*cols]);
    }


    /**
     * Creates a real dense matrix with a specified shape and fills the matrix with the specified value.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols, double value) {
        super(new Shape(rows, cols), new double[rows*cols]);
        Arrays.fill(super.entries, value);
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(double[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                Arrays.stream(entries)
                .flatMapToDouble(Arrays::stream)
                .toArray());
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(int[][] entries) {
        super(new Shape(entries.length, entries[0].length), new double[entries.length*entries[0].length]);

        // Copy the int array
        int index=0;
        for(int[] row : entries) {
            for(int value : row) {
                super.entries[index++] = value;
            }
        }
    }

    /**
     * Creates a real dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public Matrix(Matrix A) {
        super(A.shape.clone(), A.entries.clone());
    }


    /**
     * Creates a real dense matrix with specified shape filled with zeros.
     * @param shape Shape of matrix.
     */
    public Matrix(Shape shape) {
        super(shape, new double[shape.totalEntries().intValue()]);
    }


    /**
     * Creates a real dense matrix with specified shape filled with a specific value.
     * @param shape Shape of matrix.
     * @param value Value to fill matrix with.
     */
    public Matrix(Shape shape, double value) {
        super(shape, new double[shape.totalEntries().intValue()]);
        Arrays.fill(super.entries, value);
    }


    /**
     * Constructs a matrix with specified shape and entries. Note, unlike other constructors, the entries parameter
     * is not copied.
     * @param shape Shape of the matrix
     * @param entries Entries of the matrix.
     */
    public Matrix(Shape shape, double[] entries) {
        super(shape, entries);
    }


//    // TODO:
    public Matrix add(Matrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        double[] sum = RealDenseOperations.add(this, B);
        return new Matrix(this.getShape(), sum);
    }
}
