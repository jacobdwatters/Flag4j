package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.MatrixBase;

/**
 * Complex dense matrix. Stored in row major format.
 */
public class CMatrix extends MatrixBase<CNumber[]> {


    /**
     * Constructs a square complex dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size) {
        super(new Shape(size, size), new CNumber[size*size]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = new CNumber();
        }
    }


    /**
     * Creates a square complex dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size, double value) {
        super(new Shape(size, size), new CNumber[size*size]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = new CNumber(value);
        }
    }


    /**
     * Creates a square complex dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size, CNumber value) {
        super(new Shape(size, size), new CNumber[size*size]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = value.clone();
        }
    }


    /**
     * Creates a complex dense matrix of a specified shape filled with zeros.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @throws IllegalArgumentException if either rows or cols is negative.
     */
    public CMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new CNumber[rows*cols]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = new CNumber();
        }
    }


    /**
     * Creates a complex dense matrix with a specified shape and fills the matrix with the specified value.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either rows or cols is negative.
     */
    public CMatrix(int rows, int cols, double value) {
        super(new Shape(rows, cols), new CNumber[rows*cols]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = new CNumber(value);
        }
    }


    /**
     * Creates a complex dense matrix with a specified shape and fills the matrix with the specified value.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either rows or cols is negative.
     */
    public CMatrix(int rows, int cols, CNumber value) {
        super(new Shape(rows, cols), new CNumber[rows*cols]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = value.clone();
        }
    }


    /**
     * Creates a complex dense matrix with specified shape.
     * @param shape Shape of the matrix.
     */
    public CMatrix(Shape shape) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = new CNumber();
        }
    }


    /**
     * Creates a complex dense matrix with specified shape filled with specified value.
     * @param shape Shape of the matrix.
     * @param value Value to fill matrix with.
     */
    public CMatrix(Shape shape, double value) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = new CNumber(value);
        }
    }


    /**
     * Creates a complex dense matrix with specified shape filled with specified value.
     * @param shape Shape of the matrix.
     * @param value Value to fill matrix with.
     */
    public CMatrix(Shape shape, CNumber value) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = value.clone();
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(String[][] entries) {
        super(new Shape(entries.length, entries[0].length), new CNumber[entries.length*entries[0].length]);

        // Copy the string array
        int index=0;
        for(String[] row : entries) {
            for(String value : row) {
                super.entries[index++] = new CNumber(value);
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(CNumber[][] entries) {
        super(new Shape(entries.length, entries[0].length), new CNumber[entries.length*entries[0].length]);

        // Copy the string array
        int index=0;
        for(CNumber[] row : entries) {
            for(CNumber value : row) {
                super.entries[index++] = value.clone();
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(double[][] entries) {
        super(new Shape(entries.length, entries[0].length), new CNumber[entries.length*entries[0].length]);

        // Copy the double array
        int index=0;
        for(double[] row : entries) {
            for(double value : row) {
                super.entries[index++] = new CNumber(value);
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(int[][] entries) {
        super(new Shape(entries.length, entries[0].length), new CNumber[entries.length*entries[0].length]);

        // Copy the int array
        int index=0;
        for(int[] row : entries) {
            for(int value : row) {
                super.entries[index++] = new CNumber(value);
            }
        }
    }


    /**
     * Creates a complex dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(Matrix A) {
        super(A.shape.clone(), new CNumber[A.totalEntries().intValue()]);

        for(int i=0; i<super.entries.length; i++) {
            super.entries[i] = new CNumber(A.entries[i]);
        }
    }


    /**
     * Creates a complex dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(CMatrix A) {
        super(A.shape.clone(), new CNumber[A.totalEntries().intValue()]);

        for(int i=0; i<super.entries.length; i++) {
            super.entries[i] = A.entries[i].clone();
        }
    }


    /**
     * Constructs a complex matrix with specified shapes and entries. Note, unlike other constructors, the entries parameter
     * is not copied.
     * @param shape Shape of the matrix.
     * @param entries Entries of the matrix.
     */
    public CMatrix(Shape shape, CNumber[] entries) {
        super(shape, entries);
    }
}
