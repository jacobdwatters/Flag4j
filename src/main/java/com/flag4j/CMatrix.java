/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.ComplexMatrixBase;
import com.flag4j.util.ArrayUtils;

import java.util.Arrays;

/**
 * Complex dense matrix. Stored in row major format.
 */
public class CMatrix extends ComplexMatrixBase {


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
            super.entries[i] = value.copy();
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
            super.entries[i] = value.copy();
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
            super.entries[i] = value.copy();
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
                super.entries[index++] = value.copy();
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
        super(A.shape.copy(), new CNumber[A.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }


    /**
     * Creates a complex dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(CMatrix A) {
        super(A.shape.copy(), new CNumber[A.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(A.entries, super.entries);
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


    /**
     * Constructs a complex matrix with specified shapes and entries. Note, unlike other constructors, the entries parameter
     * is not copied.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @param entries Entries of the matrix.
     */
    public CMatrix(int numRows, int numCols, CNumber[] entries) {
        super(new Shape(numRows, numCols), entries);
    }


    /**
     * Converts this matrix to an equivalent real matrix. Imaginary components are ignored.
     * @return A real matrix with equivalent real parts.
     */
    @Override
    public Matrix toReal() {
        return null;
    }


    @Override
    public boolean equals(Object B) {
        boolean result;

        if(B instanceof CMatrix) {
            CMatrix mat = (CMatrix) B;
            result = Arrays.equals(this.entries, mat.entries);
        } else {
            result = false;
        }

        return result;
    }


    public CNumber get(int... indices) {
        return this.entries[this.shape.entriesIndex(indices)];
    }
}
