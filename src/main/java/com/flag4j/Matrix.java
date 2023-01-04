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
import com.flag4j.core.*;
import com.flag4j.operations.MatrixMultiply;
import com.flag4j.operations.common.real.Aggregate;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.operations.concurrency.CheckConcurrent;
import com.flag4j.operations.dense.real.*;
import com.flag4j.operations.dense.real_complex.RealComplexDenseEquals;
import com.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.Arrays;

/**
 * Real dense matrix. Stored in row major format. This class is equivalent to a real dense tensor of rank 2.
 */
public class Matrix extends RealMatrixBase implements
        MatrixComparisonsMixin<Matrix, Matrix, SparseMatrix, CMatrix, Matrix, Double>,
        MatrixManipulationsMixin<Matrix, Matrix, SparseMatrix, CMatrix, Matrix, Double>,
        MatrixOperationsMixin<Matrix, Matrix, SparseMatrix, CMatrix, Matrix, Double>,
        MatrixPropertiesMixin<Matrix, Matrix, SparseMatrix, CMatrix, Matrix, Double> {


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
    public Matrix(Double[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);

        int index = 0;

        for(Double[] row : entries) {
            for(Double value : row) {
                super.entries[index++] = value;
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(Integer[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);

        int index = 0;

        for(Integer[] row : entries) {
            for(Integer value : row) {
                super.entries[index++] = value;
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(double[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);

        int index = 0;

        for(double[] row : entries) {
            for(double value : row) {
                super.entries[index++] = value;
            }
        }
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
     * Constructs a matrix with specified shape and entries. Note, unlike other constructors, the entries' parameter
     * is not copied.
     * @param shape Shape of the matrix
     * @param entries Entries of the matrix.
     */
    public Matrix(Shape shape, double[] entries) {
        super(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent complex matrix.
     *
     * @return A complex matrix with equivalent real part and zero imaginary part.
     */
    @Override
    public CMatrix toComplex() {
        return new CMatrix(this);
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        if(isSquare()) {
            for(int i=0; i<numRows; i++) {
                for(int j=0; j<numCols; j++) {
                    if(i==j && entries[i*numCols + j]!=1) {
                        return false; // No need to continue
                    } else if(i!=j && entries[i*numCols + j]!=0) {
                        return false; // No need to continue
                    }
                }
            }

        } else {
            // An identity matrix must be square.
            return false;
        }

        // If we make it to this point this matrix must be an identity matrix.
        return true;
    }


    /**
     * Checks if an object is equal to this matrix object. Valid object types are: {@link Matrix}, {@link CMatrix},
     * {@link SparseMatrix}, and {@link SparseCMatrix}. These matrices are equal to this matrix if all entries are
     * numerically equal to the corresponding element of this matrix. If the matrix is complex, then the imaginary
     * component must be zero to be equal.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices are numerically equivalent and false otherwise.
     */
    public boolean equals(Object object) {
        boolean equal;

        if(object instanceof Matrix) {
            Matrix mat = (Matrix) object;
            equal = RealDenseEquals.matrixEquals(this, mat);

        } else if(object instanceof CMatrix) {
            CMatrix mat = (CMatrix) object;
            equal = RealComplexDenseEquals.matrixEquals(this, mat);

        } else if(object instanceof SparseMatrix) {
            SparseMatrix mat = (SparseMatrix) object;
            equal = RealDenseSparseEquals.matrixEquals(this, mat);

        } else if(object instanceof SparseCMatrix) {
            SparseCMatrix mat = (SparseCMatrix) object;
            equal = RealComplexDenseSparseEquals.matrixEquals(this, mat);

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Creates a hashcode for this matrix. Note, method adds {@link Arrays#hashCode(double[])} on the
     * underlying data array and the underlying shape array.
     * @return The hashcode for this matrix.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(entries)+Arrays.hashCode(shape.dims);
    }


    /**
     * Checks if matrices are inverses of each other.
     *
     * @param B Second matrix.
     * @return True if matrix B is an inverse of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    @Override
    public boolean isInv(Matrix B) {
        return false;
    }


    /**
     * Reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     * in the reshaped matrix.
     *
     * @param shape New Shape.
     * @return A matrix which is equivalent to this matrix but with the specified dimensions.
     * @throws IllegalArgumentException If either,<br>
     *                                  - The shape array contains negative indices.<br>
     *                                  - This matrix cannot be reshaped to the specified dimensions.
     */
    @Override
    public Matrix reshape(Shape shape) {
        // Ensure the total number of entries in each shape is equal
        ParameterChecks.assertBroadcastable(shape, this.shape);
        return new Matrix(shape, entries.clone());
    }


    /**
     * Reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     * in the reshaped matrix.
     *
     * @param numRows The number of rows in the reshaped matrix.
     * @param numCols The number of columns in the reshaped matrix.
     * @return A matrix which is equivalent to this matrix but with the specified dimensions.
     */
    @Override
    public Matrix reshape(int numRows, int numCols) {
        return reshape(new Shape(numRows, numCols));
    }


    /**
     * Flattens a matrix to have a single row. To flatten matrix to a single column, see {@link #flatten(int)}.
     *
     * @return The flat version of this matrix.
     */
    @Override
    public Matrix flatten() {
        return reshape(new Shape(1, entries.length));
    }


    /**
     * Flattens a matrix along a specified axis. Also see {@link #flatten()}.
     *
     * @param axis - If axis=0, flattens to a row vector.<br>
     *             - If axis=1, flattens to a column vector.
     * @return The flat version of this matrix.
     */
    @Override
    public Matrix flatten(int axis) {
        if(axis==Axis2D.row()) {
            // Flatten to single row
            return reshape(new Shape(1, entries.length));
        } else if(axis==Axis2D.col()) {
            // Flatten to single column
            return reshape(new Shape(entries.length, 1));
        } else {
            // Unknown axis
            throw new IllegalArgumentException(ErrorMessages.axisErr(axis, Axis2D.allAxes()));
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(Double[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(double[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(Integer[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(int[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(Double[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(Integer[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(double[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(int[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(Double[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols());

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(Integer[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols());

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(double[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols);

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(int[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols);

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set within this matrix.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(Matrix values, int rowStart, int colStart) {
        for(int i=0; i<values.numRows; i++) {
            for(int j=0; j<values.numCols; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] =
                        values.entries[i* values.numCols + j];
            }
        }
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(Double[][] values, int rowStart, int colStart) {
        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(Integer[][] values, int rowStart, int colStart) {
        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(double[][] values, int rowStart, int colStart) {
        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(int[][] values, int rowStart, int colStart) {
        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(Matrix values, int rowStart, int colStart) {
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.numRows; i++) {
            for(int j=0; j<values.numCols; j++) {
                copy.entries[(i+rowStart)*numCols + j+colStart] = values.entries[values.shape.entriesIndex(i, j)];
            }
        }

        return copy;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(Double[][] values, int rowStart, int colStart) {
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                copy.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return copy;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(Integer[][] values, int rowStart, int colStart) {
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                copy.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return copy;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(double[][] values, int rowStart, int colStart) {
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                copy.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return copy;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(int[][] values, int rowStart, int colStart) {
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                copy.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return copy;
    }


    /**
     * Removes a specified row from this matrix.
     * @param rowIndex Index of the row to remove from this matrix.
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRow(int rowIndex) {
        Matrix copy = new Matrix(this.numRows-1, this.numCols);

        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(i!=rowIndex) {
                for(int j=0; j<this.numCols; j++) {
                    copy.entries[row*copy.numCols + j] = this.entries[i*numCols + j];
                }
                row++;
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRows(int... rowIndices) {
        return null;
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCol(int colIndex) {
        Matrix copy = new Matrix(this.numRows, this.numCols-1);

        int col;

        for(int i=0; i<this.numRows; i++) {
            col = 0;
            for(int j=0; j<this.numCols; j++) {
                if(j!=colIndex) {
                    copy.entries[i*copy.numCols + col] = this.entries[i*numCols + j];
                    col++;
                }
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCols(int... colIndices) {
        return null;
    }


    /**
     * Rounds this matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @return A copy of this matrix with each entry rounded to the nearest whole number.
     */
    @Override
    public Matrix round() {
        return new Matrix(this.shape, RealOperations.round(this.entries));
    }


    /**
     * Rounds a matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     */
    @Override
    public Matrix round(int precision) {
        return new Matrix(this.shape, RealOperations.round(this.entries, precision));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within 1.0*E^-12 of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this matrix with rounded values.
     */
    @Override
    public Matrix roundToZero() {
        return new Matrix(this.shape, RealOperations.roundToZero(this.entries, 1.0E-12));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     */
    @Override
    public Matrix roundToZero(double threshold) {
        return new Matrix(this.shape, RealOperations.roundToZero(this.entries, threshold));
    }


    /**
     * Computes the element-wise addition between two matrices.
     *
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix add(Matrix B) {
        return new Matrix(this.shape.clone(),
                RealDenseOperations.add(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix add(double a) {
        return new Matrix(this.shape.clone(),
                RealDenseOperations.add(this.entries, a)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        return new CMatrix(this.shape.clone(),
                RealComplexDenseOperations.add(this.entries, a)
        );
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix add(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(CMatrix B) {
        return new CMatrix(this.shape.clone(),
                RealComplexDenseOperations.add(B.entries, B.shape, this.entries, this.shape)
        );
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix sub(Matrix B) {
        return new Matrix(this.shape.clone(),
                RealDenseOperations.sub(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Subtracts specified value to all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix sub(double a) {
        return new Matrix(this.shape.clone(),
                RealDenseOperations.sub(this.entries, a)
        );
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        return new CMatrix(this.shape.clone(),
                RealComplexDenseOperations.sub(this.entries, a)
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public Matrix scalMult(double factor) {
        return new Matrix(this.shape.clone(),
                RealOperations.scalMult(this.entries, factor)
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CMatrix scalMult(CNumber factor) {
        return new CMatrix(this.shape.clone(),
                RealComplexDenseOperations.scalMult(this.entries, factor)
        );
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public Matrix scalDiv(double divisor) {
        return new Matrix(this.shape.clone(),
                RealDenseOperations.scalDiv(this.entries, divisor)
        );
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scaler value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CMatrix scalDiv(CNumber divisor) {
        return new CMatrix(this.shape.clone(),
                RealComplexDenseOperations.scalDiv(this.entries, divisor)
        );
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        return Aggregate.sum(entries);
    }


    /**
     * Computes the element-wise square root of a tensor. If this matrix contains negative entries, the corresponding
     * square root will be {@code NaN}.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public Matrix sqrt() {
        return new Matrix(this.shape.clone(),
                RealOperations.sqrt(entries)
        );
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public Matrix abs() {
        return new Matrix(this.shape.clone(),
                RealOperations.abs(entries)
        );
    }


    /**
     * Computes the transpose of the matrix. Same as {@link #T()}.
     *
     * @return The transpose of this matrix.
     */
    @Override
    public Matrix transpose() {
        double[] transposeEntries;

        // TODO: Check if standard or blocked should be used.
        if(CheckConcurrent.simpleMatrixCheck(numRows, numCols)) {
            // Use concurrent implementation
            transposeEntries = RealDenseTranspose.blockedMatrixConcurrent(
                    entries, numRows, numCols
            );
        } else {
            // Use sequential implementation
            transposeEntries = RealDenseTranspose.blockedMatrix(
                    entries, numRows, numCols
            );
        }

        return new Matrix(new Shape(numCols, numRows), transposeEntries);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Matrix T() {
        return transpose();
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public Matrix recep() {
        return new Matrix(
                shape.clone(),
                RealDenseOperations.recep(entries)
        );
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix sub(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(CMatrix B) {
        return new CMatrix(
                shape.clone(),
                RealComplexDenseOperations.sub(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(Matrix B) {
        double[] entries = MatrixMultiply.dispatch(this, B);
        Shape shape = new Shape(this.numRows, B.numCols);

        return new Matrix(shape, entries);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CMatrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Matrix mult(Vector b) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Matrix mult(SparseVector b) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CMatrix mult(CVector b) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CMatrix mult(SparseCVector b) {
        return null;
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method will be significantly
     * faster.
     *
     * @param exponent The exponent in the matrix power.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     */
    @Override
    public Matrix pow(double exponent) {
        return null;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Matrix elemMult(Matrix B) {
        return new Matrix(
                shape.clone(),
                RealDenseOperations.elemMult(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public SparseMatrix elemMult(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CMatrix elemMult(CMatrix B) {
        return new CMatrix(
                shape.clone(),
                RealComplexDenseOperations.elemMult(B.entries, B.shape, entries, shape)
        );
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public SparseCMatrix elemMult(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise division between two matrices.
     *
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException      If B contains any zero entries.
     */
    @Override
    public Matrix elemDiv(Matrix B) {
        return new Matrix(
                shape.clone(),
                RealDenseOperations.elemDiv(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise division between two matrices.
     *
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException      If B contains any zero entries.
     */
    @Override
    public CMatrix elemDiv(CMatrix B) {
        return new CMatrix(
                shape.clone(),
                RealComplexDenseOperations.elemDiv(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double det() {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(Matrix B) {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CMatrix B) {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public Matrix directSum(Matrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseMatrix directSum(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CMatrix directSum(CMatrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix directSum(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public Matrix invDirectSum(Matrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseMatrix invDirectSum(SparseMatrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CMatrix invDirectSum(CMatrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix invDirectSum(SparseCMatrix B) {
        return null;
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    @Override
    public Matrix sumCols() {
        return null;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    @Override
    public Matrix sumRows() {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(Vector b) {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(SparseVector b) {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CVector b) {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(SparseCVector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(Vector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(SparseVector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CVector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(SparseCVector b) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public Matrix stack(Matrix B) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public Matrix stack(SparseMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(CMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(SparseCMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(Matrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(SparseMatrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(CMatrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(SparseCMatrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public Matrix augment(Matrix B) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public Matrix augment(SparseMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(CMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(SparseCMatrix B) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(Vector, int)} and {@link #augment(Vector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public Matrix stack(Vector b) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(SparseVector, int)} and {@link #augment(SparseVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public SparseMatrix stack(SparseVector b) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CVector, int)} and {@link #augment(CVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(CVector b) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(SparseCVector, int)} and {@link #augment(SparseCVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(SparseCVector b) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Vector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Vector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(Vector b, int axis) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(SparseVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(SparseVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public SparseMatrix stack(SparseVector b, int axis) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(CVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(CVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(CVector b, int axis) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(SparseCVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(SparseCVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(SparseCVector b, int axis) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(Vector)} and {@link #stack(Vector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public Matrix augment(Vector b) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(SparseVector)} and {@link #stack(SparseVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public SparseMatrix augment(SparseVector b) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CVector)} and {@link #stack(CVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CMatrix augment(CVector b) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(SparseCVector)} and {@link #stack(SparseCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CMatrix augment(SparseCVector b) {
        return null;
    }


    /**
     * Gets the element in this matrix at the specified indices.
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices is not two.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.assertArrayLengthsEq(indices.length, shape.getRank());
        return entries[shape.entriesIndex(indices)];
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    @Override
    public Double[] getRow(int i) {
        return new Double[0];
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public Double[] getCol(int j) {
        return new Double[0];
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #tr()}
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double trace() {
        ParameterChecks.assertSquare(this.shape);
        double sum = 0;
        int colsOffset = this.numCols+1;

        for(int i=0; i<this.numRows; i++) {
            sum += this.entries[i*colsOffset];
        }

        return sum;
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #trace()}
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double tr() {
        return trace();
    }


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    @Override
    public boolean isSquare() {
        return numRows==numCols;
    }


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     *
     * @return True if this matrix can be represented as either a row or column vector.
     */
    @Override
    public boolean isVector() {
        return numRows<=1 || numCols<=1;
    }


    /**
     * Checks what type of vector this matrix is. i.e. not a vector, a 1x1 matrix, a row vector, or a column vector.
     *
     * @return - If this matrix can not be represented as a vector, then returns -1. <br>
     * - If this matrix is a 1x1 matrix, then returns 0. <br>
     * - If this matrix is a row vector, then returns 1. <br>
     * - If this matrix is a column vector, then returns 2.
     */
    @Override
    public int vectorType() {
        return 0;
    }


    /**
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     *
     * @return True is this matrix is triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTri() {
        return false;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriL() {
        return false;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriU() {
        return false;
    }


    /**
     * Checks if this matrix is diagonal.
     *
     * @return True is this matrix is diagonal. Otherwise, returns false.
     */
    @Override
    public boolean isDiag() {
        return false;
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return false;
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     *
     * @return True if this matrix is singular. Otherwise, returns false.
     */
    @Override
    public boolean isSingular() {
        return false;
    }


    /**
     * Checks if a matrix is invertible.<br>
     * Also see {@link #isSingular()}.
     *
     * @return True if this matrix is invertible.
     */
    @Override
    public boolean isInvertible() {
        return false;
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    @Override
    public double norm(double p, double q) {
        return RealOperations.matrixNorm(entries, shape, p, q);
    }


    /**
     * Checks if a matrix is diagonalizable. A matrix is diagonalizable if and only if
     * the multiplicity for each eigenvalue is equivalent to the eigenspace for that eigenvalue.
     *
     * @return True if the matrix is diagonalizable. Otherwise, returns false.
     */
    @Override
    public boolean isDiagonalizable() {
        return false;
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return RealDenseCheckOperations.isZeros(entries);
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return RealDenseCheckOperations.isOnes(entries);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    @Override
    public void set(double value, int... indices) {
        RealDenseSetOperations.set(entries, shape, value, indices);
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double min() {
        return Aggregate.min(entries);
    }


    /**
     * Finds the maximum value in this matrix. If this matrix has zero entries, the method will return 0.
     * @return The maximum value in this matrix.
     */
    @Override
    public Double max() {
        return Aggregate.max(entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public Double minAbs() {
        return Aggregate.minAbs(entries);
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public Double maxAbs() {
        return Aggregate.maxAbs(entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return shape.getIndices(AggregateDenseReal.argMin(entries));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return shape.getIndices(AggregateDenseReal.argMax(entries));
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return RealOperations.matrixNorm(entries, shape);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    @Override
    public double norm(double p) {
        return RealOperations.matrixNorm(entries, shape, p);
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double infNorm() {
        return RealOperations.matrixInfNorm(entries, shape);
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double maxNorm() {
        return RealOperations.matrixMaxNorm(entries);
    }
}
