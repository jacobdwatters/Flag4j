/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.dense;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.complex_numbers.CNumberUtils;
import org.flag4j.core.MatrixMixin;
import org.flag4j.core.RealMatrixMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.dense_base.DenseMatrixMixin;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.operations.MatrixMultiplyDispatcher;
import org.flag4j.operations.RealDenseMatrixMultiplyDispatcher;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.common.real.RealProperties;
import org.flag4j.operations.dense.real.RealDenseDeterminant;
import org.flag4j.operations.dense.real.RealDenseEquals;
import org.flag4j.operations.dense.real.RealDenseProperties;
import org.flag4j.operations.dense.real.RealDenseSetOperations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseMatrixMultiplication;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseMatrixMultTranspose;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseMatrixOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixMultTranspose;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixOperations;
import org.flag4j.sparse.*;
import org.flag4j.util.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Real dense matrix. Stored in row major format. This class is mostly equivalent to a {@link Tensor real dense tensor}.
 * However, specialized methods are provided for this class which may result in slightly better performance than
 * equivalent operations with a {@link Tensor real dense tensor} of rank 2. Additionally, methods specific to matrices
 * which may not be available for tensors are provided.
 */
public class Matrix
        extends RealDenseTensorBase<Matrix, CMatrix>
        implements MatrixMixin<Matrix, Matrix, CooMatrix, CMatrix, Double, Vector, Vector>,
        RealMatrixMixin<Matrix, CMatrix>,
        DenseMatrixMixin<Matrix, Double> {

    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Constructs a square real dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size) {
        super(new Shape(size, size), new double[size*size]);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
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
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols) {
        super(new Shape(rows, cols), new double[rows*cols]);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
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
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(Double[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

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
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

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
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

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
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

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
        super(A.shape.copy(), A.entries.clone());
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a real dense matrix with specified shape filled with zeros.
     * @param shape Shape of matrix.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape) {
        super(shape, new double[shape.totalEntries().intValue()]);
        ParameterChecks.assertRank(2, shape);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a real dense matrix with specified shape filled with a specific value.
     * @param shape Shape of matrix.
     * @param value Value to fill matrix with.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape, double value) {
        super(shape, new double[shape.totalEntries().intValue()]);
        Arrays.fill(super.entries, value);
        ParameterChecks.assertRank(2, shape);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Constructs a matrix with specified shape and entries. Note, unlike other constructors, the entries' parameter
     * is not copied.
     * @param shape Shape of the matrix
     * @param entries Entries of the matrix.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape, double[] entries) {
        super(shape, entries);
        ParameterChecks.assertRank(2, shape);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Constructs a matrix with specified shape and entries. Note, unlike other constructors, the entries' parameter
     * is not copied.
     * @param numRows Number of rows in this matrix.
     * @param numCols Number of columns in this matrix.
     * @param entries Entries of the matrix.
     */
    public Matrix(int numRows, int numCols, double[] entries) {
        super(new Shape(numRows, numCols), entries);
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }



    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected Matrix makeTensor(Shape shape, double[] entries) {
        return new Matrix(shape, entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CMatrix makeComplexTensor(Shape shape, double[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CMatrix makeComplexTensor(Shape shape, CNumber[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Simply returns this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected Matrix getSelf() {
        return this;
    }


    /**
     * Gets the number of rows in this matrix.
     * @return The number of rows in this matrix.
     */
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     * @return The number of columns in this matrix.
     */
    public int numCols() {
        return numCols;
    }


    /**
     * Gets the shape of this matrix.
     *
     * @return The shape of this matrix.
     */
    @Override
    public Shape shape() {
        return shape;
    }


    /**
     * Converts this matrix to an equivalent complex tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public Tensor toTensor() {
        return new Tensor(this.shape.copy(), this.entries.clone());
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     * @return A vector equivalent to this matrix.
     */
    @Override
    public Vector toVector() {
        return new Vector(this.entries.clone());
    }


    /**
     * Converts this matrix to a sparse COO matrix.
     * @return A sparse coo matrix equivalent to this matrix.
     * @see #toCsr()
     */
    public CooMatrix toCoo() {
        List<Double> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;
            int stop = rowOffset + numCols;
            int j=0;

            while(rowOffset < stop) {
                double value = entries[rowOffset++];

                if(value!=0) {
                    // Then we have a non-zero value.
                    sparseEntries.add(value);
                    rowIndices.add(i);
                    colIndices.add(j);
                }

                j++;
            }
        }

        return new CooMatrix(shape.copy(), sparseEntries, rowIndices, colIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @return A sparse coo matrix equivalent to this matrix.
     * @see #toCoo()
     */
    public CsrMatrix toCsr() {
        return toCoo().toCsr();
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     * @see #isCloseToI()
     */
    @Override
    public boolean isI() {
        int pos = 0;

        if(isSquare()) {
            for(int i=0; i<numRows; i++) {
                for(int j=0; j<numCols; j++) {
                    if((i==j && entries[pos]!=1) || (i!=j && entries[pos]!=0)) {
                        return false; // No need to continue
                    }

                    pos++;
                }
            }

        } else {
            // An identity matrix must be square.
            return false;
        }

        // If we make it to this point, this matrix must be an identity matrix.
        return true;
    }


    /**
     * Checks that this matrix is close to the identity matrix according to
     * {@link RealProperties#allClose(double[], double[])}
     * @return True if this matrix is approximately the identity matrix.
     * @see #isI()
     */
    public boolean isCloseToI() {
        return RealDenseProperties.isCloseToIdentity(this);
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link Matrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(!(object instanceof Matrix)) return false;

        Matrix src2 = (Matrix) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
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
        if(axis== Axis2D.row()) {
            // Flatten to single row
            return reshape(new Shape(1, entries.length));
        } else if(axis==Axis2D.col()) {
            // Flatten to single column
            return reshape(new Shape(entries.length, 1));
        } else {
            // Unknown axis
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, Axis2D.allAxes()));
        }
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A reference to this matrix.
     */
    @Override
    public Matrix set(double value, int row, int col) {
        entries[row*numCols + col] = value;
        return this;
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A reference to this matrix.
     */
    @Override
    public Matrix set(Double value, int row, int col) {
        entries[row*numCols + col] = value;
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public Matrix setValues(Double[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public Matrix setValues(double[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    public Matrix setValues(Integer[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public Matrix setValues(int[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public Matrix setCol(Double[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }

        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public Matrix setCol(Integer[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }

        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public Matrix setCol(double[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }

        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public Matrix setCol(int[] values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numRows);

        for(int i=0; i<values.length; i++) {
            super.entries[i*numCols + colIndex] = values[i];
        }

        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the {@code values} vector has a different length than the number of rows of this matrix.
     */
    public Matrix setCol(Vector values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.size, this.numRows);

        int rowOffset = 0;
        for(int i=0; i<values.size; i++) {
            super.entries[rowOffset + colIndex] = values.entries[i];
            rowOffset += numCols;
        }

        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column. This method assumes that the indices of the sparse vector are sorted.
     *                 If this is not the case, call {@link CooVector#sortIndices()} first.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the {@code values} vector has a different length than the number of rows of this matrix.
     */
    public Matrix setCol(CooVector values, int colIndex) {
        ParameterChecks.assertArrayLengthsEq(values.size, this.numRows);

        int sparseIdx = 0;
        for(int i=0; i<values.size; i++) {
            super.entries[i*numCols + colIndex] = (i == values.indices[sparseIdx]) ?
                    values.entries[sparseIdx++] : 0;
        }

        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public Matrix setRow(Double[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols);

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }

        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public Matrix setRow(Integer[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols);

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }

        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public Matrix setRow(double[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols);
        System.arraycopy(values, 0, super.entries, rowIndex*numCols, values.length);
        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public Matrix setRow(int[] values, int rowIndex) {
        ParameterChecks.assertArrayLengthsEq(values.length, this.numCols);

        for(int i=0; i<values.length; i++) {
            super.entries[rowIndex*numCols + i] = values[i];
        }

        return this;
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public Matrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        Matrix slice = new Matrix(rowEnd-rowStart, colEnd-colStart);

        for(int i=0; i<slice.numRows; i++) {
            System.arraycopy(
                    this.entries, (i+rowStart)*this.numCols + colStart,
                    slice.entries, i*slice.numCols, slice.numCols
            );
        }

        return slice;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set within this matrix.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(Matrix values, int rowStart, int colStart) {
        ParameterChecks.assertLessEq(numRows, rowStart+values.numRows);
        ParameterChecks.assertLessEq(numCols, colStart+values.numCols);
        ParameterChecks.assertGreaterEq(0, rowStart, colStart);

        for(int i=0; i<values.numRows; i++) {
            System.arraycopy(
                    values.entries, i*values.numCols,
                    this.entries, (i+rowStart)*numCols + colStart, values.numCols
            );
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(CooMatrix values, int rowStart, int colStart) {
        ParameterChecks.assertLessEq(numRows, rowStart+values.numRows);
        ParameterChecks.assertLessEq(numCols, colStart+values.numCols);
        ParameterChecks.assertGreaterEq(0, rowStart, colStart);

        // Fill slice with zeros
        ArrayUtils.stridedFillZeros(
                this.entries,
                rowStart*this.numCols+colStart,
                values.numCols,
                this.numCols-values.numCols
        );

        // Copy sparse values
        int rowIndex, colIndex;
        for(int i=0; i<values.entries.length; i++) {
            rowIndex = values.rowIndices[i];
            colIndex = values.colIndices[i];

            this.entries[(rowIndex+rowStart)*this.numCols + colIndex + colStart] = values.entries[i];
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(Double[][] values, int rowStart, int colStart) {
        ParameterChecks.assertLessEq(numRows, rowStart+values.length);
        ParameterChecks.assertLessEq(numCols, colStart+values[0].length);
        ParameterChecks.assertGreaterEq(0, rowStart, colStart);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(Integer[][] values, int rowStart, int colStart) {
        ParameterChecks.assertLessEq(numRows, rowStart+values.length);
        ParameterChecks.assertLessEq(numCols, colStart+values[0].length);
        ParameterChecks.assertGreaterEq(0, rowStart, colStart);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(double[][] values, int rowStart, int colStart) {
        ParameterChecks.assertLessEq(numRows, rowStart+values.length);
        ParameterChecks.assertLessEq(numCols, colStart+values[0].length);
        ParameterChecks.assertGreaterEq(0, rowStart, colStart);

        for(int i=0; i<values.length; i++) {
            System.arraycopy(
                    values[i], 0,
                    this.entries, (i+rowStart)*numCols + colStart, values[0].length
            );
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(int[][] values, int rowStart, int colStart) {
        ParameterChecks.assertLessEq(numRows, rowStart+values.length);
        ParameterChecks.assertLessEq(numCols, colStart+values[0].length);
        ParameterChecks.assertGreaterEq(0, rowStart, colStart);

        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values[0].length; j++) {
                this.entries[(i+rowStart)*numCols + j+colStart] = values[i][j];
            }
        }

        return this;
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
            System.arraycopy(
                    values.entries, i*values.numCols,
                    copy.entries, (i+rowStart)*numCols + colStart, values.numCols
            );
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
    public Matrix setSliceCopy(CooMatrix values, int rowStart, int colStart) {
        Matrix copy = this.copy();

        // Fill slice with zeros
        ArrayUtils.stridedFillZeros(
                copy.entries,
                rowStart*copy.numCols+colStart,
                values.numCols,
                copy.numCols-values.numCols
        );

        // Copy sparse values
        int rowIndex, colIndex;
        for(int i=0; i<values.entries.length; i++) {
            rowIndex = values.rowIndices[i];
            colIndex = values.colIndices[i];

            copy.entries[(rowIndex+rowStart)*copy.numCols + colIndex + colStart] = values.entries[i];
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
                System.arraycopy(this.entries, i*numCols, copy.entries, row*copy.numCols, this.numCols);
                row++;
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRows(int... rowIndices) {
        Matrix copy = new Matrix(this.numRows-rowIndices.length, this.numCols);

        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(ArrayUtils.notContains(rowIndices, i)) {
                System.arraycopy(this.entries, i*numCols, copy.entries, row*copy.numCols, this.numCols);
                row++;
            }
        }

        return copy;
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
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCols(int... colIndices) {
        Matrix copy = new Matrix(this.numRows, this.numCols-colIndices.length);

        int col;

        for(int i=0; i<this.numRows; i++) {
            col = 0;
            for(int j=0; j<this.numCols; j++) {
                if(ArrayUtils.notContains(colIndices, j)) {
                    copy.entries[i*copy.numCols + col] = this.entries[i*numCols + j];
                    col++;
                }
            }
        }

        return copy;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix add(CooMatrix B) {
        return RealDenseSparseMatrixOperations.add(this, B);
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
        return new CMatrix(this.shape.copy(),
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
    public CMatrix add(CooCMatrix B) {
        return RealComplexDenseSparseMatrixOperations.add(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(CooMatrix B) {
        RealDenseSparseMatrixOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(CooMatrix B) {
        RealDenseSparseMatrixOperations.addEq(this, B);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Matrix T() {
        return TransposeDispatcher.dispatch(this);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix sub(CooMatrix B) {
        return RealDenseSparseMatrixOperations.sub(this, B);
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
                shape.copy(),
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
    public CMatrix sub(CooCMatrix B) {
        return RealComplexDenseSparseMatrixOperations.sub(this, B);
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
        double[] entries = RealDenseMatrixMultiplyDispatcher.dispatch(this, B);
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
    public Matrix mult(CooMatrix B) {
        ParameterChecks.assertMatMultShapes(this.shape, B.shape);
        double[] entries = RealDenseSparseMatrixMultiplication.standard(
                this.entries, this.shape, B.entries, B.rowIndices, B.colIndices, B.shape
        );
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
    public CMatrix mult(CMatrix B) {
        CNumber[] entries = MatrixMultiplyDispatcher.dispatch(this, B);
        Shape shape = new Shape(this.numRows, B.numCols);

        return new CMatrix(shape, entries);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CooCMatrix B) {
        ParameterChecks.assertMatMultShapes(this.shape, B.shape);
        CNumber[] entries = RealComplexDenseSparseMatrixMultiplication.standard(
                this.entries, this.shape, B.entries, B.rowIndices, B.colIndices, B.shape
        );
        Shape shape = new Shape(this.numRows, B.numCols);

        return new CMatrix(shape, entries);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Vector mult(Vector b) {
        ParameterChecks.assertMatMultShapes(this.shape, new Shape(b.size, 1));
        double[] entries = MatrixMultiplyDispatcher.dispatch(this, b);
        return new Vector(entries);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Vector mult(CooVector b) {
        ParameterChecks.assertMatMultShapes(this.shape, new Shape(b.size, 1));
        double[] entries = RealDenseSparseMatrixMultiplication.standardVector(
                this.entries, this.shape, b.entries, b.indices
        );

        return new Vector(entries);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(CVector b) {
        ParameterChecks.assertMatMultShapes(this.shape, new Shape(b.size, 1));
        CNumber[] entries = RealComplexDenseMatrixMultiplication.standardVector(
                this.entries, this.shape, b.entries, b.shape
        );

        return new CVector(entries);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(CooCVector b) {
        ParameterChecks.assertMatMultShapes(this.shape, new Shape(b.size, 1));
        CNumber[] entries = RealComplexDenseSparseMatrixMultiplication.standardVector(
                this.entries, this.shape, b.entries, b.indices
        );

        return new CVector(entries);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.}{@link #mult(Matrix) mult}{@code (B.}{@link #T() T}{@code ())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(Matrix B) {
        // Ensure this matrix can be multiplied to the transpose of B.
        ParameterChecks.assertEquals(this.numCols, B.numCols);

        return new Matrix(
                new Shape(this.numRows, B.numRows),
                RealDenseMatrixMultiplyDispatcher.dispatchTranspose(this, B)
        );
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.}{@link #mult(CooMatrix) mult}{@code (B.}{@link #T() T}{@code ())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(CooMatrix B) {
        // Ensure this matrix can be multiplied to the transpose of B.
        ParameterChecks.assertEquals(this.numCols, B.numCols);

        return new Matrix(
                new Shape(this.numRows, B.numRows),
                RealDenseSparseMatrixMultTranspose.multTranspose(
                        this.entries, this.shape, B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.}{@link #mult(CMatrix) mult}{@code (B.}{@link #T() T}{@code ())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CMatrix B) {
        // Ensure this matrix can be multiplied to the transpose of B.
        ParameterChecks.assertEquals(this.numCols, B.numCols);

        return new CMatrix(
                new Shape(this.numRows, B.numRows),
                MatrixMultiplyDispatcher.dispatchTranspose(this, B)
        );
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.}{@link #mult(CooCMatrix) mult}{@code (B.}{@link #T() T}{@code ())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CooCMatrix B) {
        // Ensure this matrix can be multiplied to the transpose of B.
        ParameterChecks.assertEquals(this.numCols, B.numCols);

        // TODO: Add dispatcher for various methods in RealComplexDenseSparseMatrixMultTranspose
        return new CMatrix(
                new Shape(this.numRows, B.numRows),
                RealComplexDenseSparseMatrixMultTranspose.multTranspose(
                        this.entries, this.shape, B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method will be significantly
     * faster.
     *
     * @param exponent The exponent in the matrix power.
     * @return The result of multiplying this matrix with itself 'exponent' times. If the exponent is zero, then the
     * identity matrix is returned.
     * @throws IllegalArgumentException If this matrix is not square or if exponent is negative.
     */
    @Override
    public Matrix pow(int exponent) {
        ParameterChecks.assertGreaterEq(0, exponent);
        ParameterChecks.assertSquareMatrix(this.shape);
        Matrix result;

        if(exponent==0) {
            result = I(this.shape);
        } else {
            result = new Matrix(this);

            for(int i=1; i<exponent; i++) {
                result = result.mult(this);
            }
        }

        return result;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CooMatrix elemMult(CooMatrix B) {
        return RealDenseSparseMatrixOperations.elemMult(this, B);
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
                shape.copy(),
                RealComplexDenseElemMult.dispatch(B.entries, B.shape, entries, shape)
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
    public CooCMatrix elemMult(CooCMatrix B) {
        return RealComplexDenseSparseMatrixOperations.elemMult(this, B);
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
                shape.copy(),
                RealComplexDenseElemDiv.dispatch(entries, shape, B.entries, B.shape)
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
        return RealDenseDeterminant.det(this);
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
        ParameterChecks.assertEqualShape(this.shape, B.shape);
        return this.T().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(CooMatrix B) {
        ParameterChecks.assertEqualShape(this.shape, B.shape);
        return this.T().mult(B).tr();
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
        ParameterChecks.assertEqualShape(this.shape, B.shape);
        return this.T().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CooCMatrix B) {
        ParameterChecks.assertEqualShape(this.shape, B.shape);
        return this.T().mult(B).tr();
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public Matrix directSum(Matrix B) {
        Matrix sum = new Matrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            System.arraycopy(entries, i*numCols, sum.entries, i*sum.numCols, this.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, (i + numRows)*sum.numCols + numCols, B.numCols);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public Matrix directSum(CooMatrix B) {
        Matrix sum = new Matrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            System.arraycopy(entries, i*numCols, sum.entries, i*sum.numCols, this.numCols);
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[(row+numRows)*sum.numCols + (col+numCols)] = B.entries[i];
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CMatrix directSum(CMatrix B) {
        CMatrix sum = new CMatrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            for(int j=0; j<this.numCols; j++) {
                sum.entries[i*sum.numCols + j] = new CNumber(entries[i*numCols + j]);
            }
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, (i+numRows)*sum.numCols+(numCols), B.numCols);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CMatrix directSum(CooCMatrix B) {
        CMatrix sum = new CMatrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            for(int j=0; j<this.numCols; j++) {
                sum.entries[i*sum.numCols + j] = new CNumber(entries[i*numCols + j]);
            }
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[(row+numRows)*sum.numCols + (col+numCols)] = B.entries[i].copy();
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public Matrix invDirectSum(Matrix B) {
        Matrix sum = new Matrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            System.arraycopy(entries, i*numCols, sum.entries, (i+B.numRows)*sum.numCols, this.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, i*sum.numCols+this.numCols, B.numCols);
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public Matrix invDirectSum(CooMatrix B) {
        Matrix sum = new Matrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            System.arraycopy(entries, i*numCols, sum.entries, (i+B.numRows)*sum.numCols, this.numCols);
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[row*sum.numCols + col + this.numCols] = B.entries[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CMatrix invDirectSum(CMatrix B) {
        CMatrix sum = new CMatrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            for(int j=0; j<this.numCols; j++) {
                sum.entries[(i+B.numRows)*sum.numCols + j] = new CNumber(entries[i*numCols + j]);
            }
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                sum.entries[i*sum.numCols + j + this.numCols] = B.entries[i*B.numCols + j].copy();
            }
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CMatrix invDirectSum(CooCMatrix B) {
        CMatrix sum = new CMatrix(this.numRows+B.numRows, this.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<this.numRows; i++) {
            for(int j=0; j<this.numCols; j++) {
                sum.entries[(i+B.numRows)*sum.numCols + j] = new CNumber(entries[i*numCols + j]);
            }
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[row*sum.numCols + col + this.numCols] = B.entries[i].copy();
        }

        return sum;
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    @Override
    public Matrix sumCols() {
        Matrix sum = new Matrix(this.numRows, 1);

        for(int i=0; i<this.numRows; i++) {
            for(int j=0; j<this.numCols; j++) {
                sum.entries[i] += this.entries[i*numCols + j];
            }
        }

        return sum;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    @Override
    public Matrix sumRows() {
        Matrix sum = new Matrix(1, this.numCols);

        for(int i=0; i<this.numRows; i++) {
            for(int j=0; j<this.numCols; j++) {
                sum.entries[j] += this.entries[i*numCols + j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     * @throws IllegalArgumentException If the vector has a different number of entries as rows in the matrix.
     */
    @Override
    public Matrix addToEachCol(Vector b) {
        ParameterChecks.assertArrayLengthsEq(numRows, b.size);
        Matrix sum = new Matrix(this);

        for(int i=0; i<sum.numRows; i++) {
            for(int j=0; j<sum.numCols; j++) {
                sum.entries[i*sum.numCols + j] += b.entries[i];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(CooVector b) {
        ParameterChecks.assertArrayLengthsEq(numRows, b.size);
        Matrix sum = new Matrix(this);

        int index;

        for(int i=0; i<b.nonZeroEntries(); i++) {
            index = b.indices[i];

            for(int j=0; j<sum.numCols; j++) {
                sum.entries[index*sum.numCols + j] += b.entries[i];
            }
        }

        return sum;
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
        ParameterChecks.assertArrayLengthsEq(numRows, b.size);
        CMatrix sum = new CMatrix(this);

        for(int i=0; i<sum.numRows; i++) {
            for(int j=0; j<sum.numCols; j++) {
                sum.entries[i*sum.numCols + j].addEq(b.entries[i]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CooCVector b) {
        ParameterChecks.assertArrayLengthsEq(numRows, b.size);
        CMatrix sum = new CMatrix(this);

        int index;

        for(int i=0; i<b.nonZeroEntries(); i++) {
            index = b.indices[i];

            for(int j=0; j<sum.numCols; j++) {
                sum.entries[index*sum.numCols + j].addEq(b.entries[i]);
            }
        }

        return sum;
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
        ParameterChecks.assertArrayLengthsEq(numCols, b.size);
        Matrix sum = new Matrix(this);

        for(int i=0; i<sum.numRows; i++) {
            for(int j=0; j<sum.numCols; j++) {
                sum.entries[i*sum.numCols + j] += b.entries[j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(CooVector b) {
        ParameterChecks.assertArrayLengthsEq(numCols, b.size);
        Matrix sum = new Matrix(this);

        int col;
        for(int i=0; i<sum.numRows; i++) {
            for(int j=0; j<b.nonZeroEntries(); j++) {
                col = b.indices[j];
                sum.entries[i*sum.numCols + col] += b.entries[j];
            }
        }

        return sum;
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
        ParameterChecks.assertArrayLengthsEq(numCols, b.size);
        CMatrix sum = new CMatrix(this);

        for(int i=0; i<sum.numRows; i++) {
            for(int j=0; j<sum.numCols; j++) {
                sum.entries[i*sum.numCols + j].addEq(b.entries[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CooCVector b) {
        ParameterChecks.assertArrayLengthsEq(numCols, b.size);
        CMatrix sum = new CMatrix(this);

        int col;
        for(int i=0; i<sum.numRows; i++) {
            for(int j=0; j<b.nonZeroEntries(); j++) {
                col = b.indices[j];
                sum.entries[i*sum.numCols + col].addEq(b.entries[j]);
            }
        }

        return sum;
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
        ParameterChecks.assertArrayLengthsEq(this.numCols, B.numCols);
        Matrix stacked = new Matrix(new Shape(this.numRows + B.numRows, this.numCols));

        System.arraycopy(this.entries, 0, stacked.entries, 0, this.entries.length);
        System.arraycopy(B.entries, 0, stacked.entries, this.entries.length, B.entries.length);

        return stacked;
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
    public Matrix stack(CooMatrix B) {
        ParameterChecks.assertArrayLengthsEq(this.numCols, B.numCols);
        Matrix stacked = new Matrix(new Shape(this.numRows + B.numRows, this.numCols));

        System.arraycopy(this.entries, 0, stacked.entries, 0, this.entries.length);

        int row, col;

        for(int i=0; i<B.entries.length; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];
            // Offset the row index for destination matrix. i.e. row+this.numRows
            stacked.entries[(row+this.numRows)*stacked.numCols + col] = B.entries[i];
        }

        return stacked;
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
        ParameterChecks.assertArrayLengthsEq(numCols, B.numCols);
        CMatrix stacked = new CMatrix(new Shape(numRows+B.numRows, numCols));

        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                stacked.entries[i*stacked.numCols + j].re = entries[i*numCols+j];
            }
        }

        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                stacked.entries[(i + numRows)*stacked.numCols + j] = B.entries[i*B.numCols+j].copy();
            }
        }

        return stacked;
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
    public CMatrix stack(CooCMatrix B) {
        ParameterChecks.assertArrayLengthsEq(this.numCols, B.numCols);
        CMatrix stacked = new CMatrix(new Shape(this.numRows + B.numRows, this.numCols));

        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                stacked.entries[i*stacked.numCols + j].re = entries[i*numCols+j];
            }
        }

        int row, col;
        for(int i=0; i<B.entries.length; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];
            // Offset the row index for destination matrix. i.e. row+this.numRows
            stacked.entries[(row+this.numRows)*stacked.numCols + col] = B.entries[i].copy();
        }

        return stacked;
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
        ParameterChecks.assertArrayLengthsEq(numRows, B.numRows);
        Matrix augmented = new Matrix(new Shape(numRows, numCols+B.numCols));

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augmented.entries, i*augmented.numCols, numCols);
        }

        // Copy entries from the B matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                augmented.entries[i*augmented.numCols + j + numCols] = B.entries[i*B.numCols + j];
            }
        }

        return augmented;
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
    public Matrix augment(CooMatrix B) {
        ParameterChecks.assertArrayLengthsEq(this.numRows, B.numRows);
        Matrix augmented = new Matrix(new Shape(this.numRows, this.numCols+B.numCols));

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augmented.entries, i*augmented.numCols, numCols);
        }

        int row, col;
        for(int i=0; i<B.entries.length; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];
            // Offset the col index for destination matrix. i.e. col+this.numCols
            augmented.entries[row*augmented.numCols + (col + numCols)] = B.entries[i];
        }

        return augmented;
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
        ParameterChecks.assertArrayLengthsEq(numRows, B.numRows);
        CMatrix augmented = new CMatrix(new Shape(numRows, numCols+B.numCols));

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                augmented.entries[i*augmented.numCols + j].re = entries[i*numCols + j];
            }
        }

        // Copy entries from the B matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                augmented.entries[i*augmented.numCols + j + numCols] = B.entries[i*B.numCols + j].copy();
            }
        }

        return augmented;
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
    public CMatrix augment(CooCMatrix B) {
        ParameterChecks.assertArrayLengthsEq(this.numRows, B.numRows);
        CMatrix augmented = new CMatrix(new Shape(this.numRows, this.numCols+B.numCols));

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                augmented.entries[i*augmented.numCols + j].re = entries[i*numCols + j];
            }
        }

        int row, col;
        for(int i=0; i<B.entries.length; i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];
            // Offset the col index for destination matrix. i.e. col+this.numCols
            augmented.entries[row*augmented.numCols + (col + numCols)] = B.entries[i].copy();
        }

        return augmented;
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
        ParameterChecks.assertArrayLengthsEq(this.numCols, b.entries.length);
        Matrix stacked = new Matrix(this.numRows+1, this.numCols);

        System.arraycopy(this.entries, 0, stacked.entries, 0, this.entries.length);
        System.arraycopy(b.entries, 0, stacked.entries, this.entries.length, b.entries.length);

        return stacked;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooVector, int)} and {@link #augment(CooVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public Matrix stack(CooVector b) {
        ParameterChecks.assertArrayLengthsEq(this.numCols, b.totalEntries().intValue());
        Matrix stacked = new Matrix(this.numRows+1, this.numCols);

        System.arraycopy(this.entries, 0, stacked.entries, 0, this.entries.length);

        int index;

        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[(stacked.numRows-1)*numCols + index] = b.entries[i];
        }

        return stacked;
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
        ParameterChecks.assertArrayLengthsEq(this.numCols, b.entries.length);
        CMatrix stacked = new CMatrix(this.numRows+1, this.numCols);

        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                stacked.entries[i*stacked.numCols + j].re = entries[i*numCols+j];
            }
        }

        for(int i=0; i<b.entries.length; i++) {
            stacked.entries[(stacked.numRows-1)*numCols + i] = b.entries[i].copy();
        }

        return stacked;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooCVector, int)} and {@link #augment(CooCVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(CooCVector b) {
        ParameterChecks.assertArrayLengthsEq(this.numCols, b.totalEntries().intValue());
        CMatrix stacked = new CMatrix(this.numRows+1, this.numCols);

        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                stacked.entries[i*stacked.numCols + j].re = entries[i*numCols+j];
            }
        }

        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[(stacked.numRows-1)*numCols + index] = b.entries[i];
        }

        return stacked;
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
        ParameterChecks.assertArrayLengthsEq(numRows, b.entries.length);
        Matrix stacked = new Matrix(numRows, numCols+1);

        // Copy elements of this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, stacked.entries, i*stacked.numCols, numCols);
        }

        // Copy elements from b vector.
        for(int i=0; i<b.entries.length; i++) {
            stacked.entries[i*stacked.numCols + stacked.numCols-1] = b.entries[i];
        }

        return stacked;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooVector)} and {@link #stack(CooVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public Matrix augment(CooVector b) {
        ParameterChecks.assertArrayLengthsEq(numRows, b.totalEntries().intValue());
        Matrix stacked = new Matrix(numRows, numCols+1);

        // Copy elements of this matrix.
        for(int i=0; i<numRows; i++) {
            if (numCols >= 0)
                System.arraycopy(entries, i*numCols, stacked.entries, i*stacked.numCols, numCols);
        }

        // Copy elements from b vector.
        for(int i=0; i<b.entries.length; i++) {
            stacked.entries[b.indices[i]*stacked.numCols + stacked.numCols-1] = b.entries[i];
        }

        return stacked;
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
        ParameterChecks.assertArrayLengthsEq(numRows, b.entries.length);
        CMatrix stacked = new CMatrix(numRows, numCols+1);

        // Copy elements of this matrix.
        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                stacked.entries[i*stacked.numCols + j].re = entries[i*numCols+j];
            }
        }

        // Copy elements from b vector.
        for(int i=0; i<b.entries.length; i++) {
            stacked.entries[i*stacked.numCols + stacked.numCols-1] = b.entries[i].copy();
        }

        return stacked;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooCVector)} and {@link #stack(CooCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CMatrix augment(CooCVector b) {
        ParameterChecks.assertArrayLengthsEq(numRows, b.totalEntries().intValue());
        CMatrix stacked = new CMatrix(numRows, numCols+1);

        // Copy elements of this matrix.
        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                stacked.entries[i*stacked.numCols + j].re = entries[i*numCols+j];
            }
        }

        // Copy elements from b vector.
        for(int i=0; i<b.entries.length; i++) {
            stacked.entries[b.indices[i]*stacked.numCols + stacked.numCols-1] = b.entries[i].copy();
        }

        return stacked;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     * @return The specified row of this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     * the number of rows in this matrix.
     */
    @Override
    public Vector getRow(int rowIdx) {
        int start = rowIdx*numCols;
        int stop = start+numCols;

        double[] row = Arrays.copyOfRange(this.entries, start, stop);

        return new Vector(row);
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     * @return The specified row of this matrix as a vector.
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     * the number of rows in this matrix.
     */
    public Vector getRowAsVector(int rowIdx) {
        int start = rowIdx*numCols;
        int stop = start+numCols;
        return new Vector(Arrays.copyOfRange(this.entries, start, stop));
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     * @return The specified column of this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     * the number of columns in this matrix.
     */
    @Override
    public Vector getCol(int colIdx) {
        double[] col = new double[numRows];

        for(int i=0; i<numRows; i++) {
            col[i] = entries[i*numCols + colIdx];
        }

        return new Vector(col);
    }


    /**
     * Get a specified column of this matrix at and below a specified row.
     *
     * @param rowStart Index of the row to begin at.
     * @param colIdx Index of column to get.
     * @return The specified column of this matrix beginning at the specified row.
     * @throws NegativeArraySizeException If {@code rowStart} is larger than the number of rows in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowStart} or {@code colIdx} is outside the bounds of this matrix.
     */
    @Override
    public Vector getColBelow(int rowStart, int colIdx) {
        double[] col = new double[numRows-rowStart];

        for(int i=rowStart; i<numRows; i++) {
            col[i-rowStart] = entries[i*numCols + colIdx];
        }

        return new Vector(col);
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     * @param colIdx Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     * @throws IllegalArgumentException If {@code rowStart} is less than 0.
     * @throws NegativeArraySizeException If {@code rowEnd} is less than {@code rowStart}.
     */
    public Vector getCol(int colIdx, int rowStart, int rowEnd) {
        ParameterChecks.assertGreaterEq(0, rowStart);
        double[] col = new double[rowEnd-rowStart];

        for(int i=rowStart; i<rowEnd; i++) {
            col[i-rowStart] = entries[i*numCols + colIdx];
        }

        return new Vector(col);
    }


    /**
     * Get a specified row of this matrix at and after a specified column.
     *
     * @param colStart Index of the row to begin at.
     * @param rowIdx Index of the row to get.
     * @return The specified row of this matrix beginning at the specified column.
     * @throws NegativeArraySizeException If {@code colStart} is larger than the number of columns in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} or {@code colStart} is outside the bounds of this matrix.
     */
    @Override
    public Vector getRowAfter(int colStart, int rowIdx) {
        if(rowIdx > this.numRows ||  rowIdx < 0 || colStart > this.numCols || colStart < 0) {
            throw new ArrayIndexOutOfBoundsException(String.format("Index (%d, %d) not in matrix.", rowIdx, colStart));
        }

        double[] row = Arrays.copyOfRange(this.entries, rowIdx*this.numCols + colStart, (rowIdx+1)*this.numCols);
        return new Vector(row);
    }


    // TODO: Pull row/colAsVector methods up to matrix operations interface.
    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     * @return The specified column of this matrix as a vector.
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     * the number of rows in this matrix.
     */
    public Vector getColAsVector(int colIdx) {
        double[] col = new double[numRows];

        for(int i=0; i<numRows; i++) {
            col[i] = entries[i*numCols + colIdx];
        }

        return new Vector(col);
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
        ParameterChecks.assertSquareMatrix(this.shape);
        double sum = 0;
        int colsOffset = this.numCols+1;

        for(int i=0; i<this.numRows; i++) {
            sum += this.entries[i*colsOffset];
        }

        return sum;
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double tr() {
        return trace();
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public Vector getDiag() {
        final int newSize = Math.min(numRows, numCols);
        double[] diag = new double[newSize];

        int idx = 0;
        for(int i=0; i<newSize; i++) {
            diag[i] = this.entries[idx];
            idx += numCols + 1;
        }

        return new Vector(diag);
    }


    /**
     * Compute the transpose of this matrix. That is, the complex conjugate transpose of this matrix. Since this is
     * a real matrix, this is equivalent to the {@link #T standard transpose}.
     *
     * @return The conjugate transpose of this matrix.
     */
    @Override
    public Matrix H() {
        return T();
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape)
     * @see #I(int, int)
     */
    public static Matrix I(int size) {
        return I(size, size);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int)
     * @see #I(Shape)
     */
    public static Matrix I(int numRows, int numCols) {
        ParameterChecks.assertGreaterEq(1, numRows, numCols);
        Matrix I = new Matrix(numRows, numCols);
        int stop = Math.min(numRows, numCols);

        for(int i=0; i<stop; i++) {
            I.entries[i*numCols+i] = 1;
        }

        return I;
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape Shape of the identity-like matrix.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified shape is not rank 2.
     * @see #I(int)
     * @see #I(int, int)
     */
    public static Matrix I(Shape shape) {
        ParameterChecks.assertRank(2, shape);
        return I(shape.get(0), shape.get(1));
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
        int type;

        if(numRows==1 || numCols==1) {
            if(numRows==1 && numCols==1) {
                type = 0;
            } else if(numRows==1) {
                type = 1;
            } else {
                // Then this matrix is equivalent to a column vector.
                type = 2;
            }
        } else {
            // Then this matrix is not equivalent to any vector.
            type = -1;
        }

        return type;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        boolean result = isSquare();

        if(result) {
            // Ensure upper half is zeros.
            for(int i=0; i<numRows; i++) {
                for(int j=i+1; j<numCols; j++) {
                    if(entries[i*numCols + j] != 0) {
                        return false; // No need to continue.
                    }
                }
            }
        }

        return result;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        boolean result = isSquare();

        if(result) {
            // Ensure lower half is zeros.
            for(int i=1; i<numRows; i++) {
                for(int j=0; j<i; j++) {
                    if(entries[i*numCols + j] != 0) {
                        return false; // No need to continue.
                    }
                }
            }
        }

        return result;
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return matrixRank() == Math.min(numRows, numCols);
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     *
     * @return True if this matrix is singular or non-square. Otherwise, returns false.
     */
    @Override
    public boolean isSingular() {
        boolean result = true;

        if(isSquare()) {
            double tol = 1.0E-16; // Tolerance for determining if determinant is zero.
            double det = RealDenseDeterminant.det(this);

            result = Math.abs(det) < tol;
        }

        return result;
    }


    /**
     * Swaps specified rows in the matrix. This is done in place.
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     * @return A reference to this matrix.
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public Matrix swapRows(int rowIndex1, int rowIndex2) {
        ParameterChecks.assertGreaterEq(0, rowIndex1, rowIndex2);
        ParameterChecks.assertGreaterEq(rowIndex1, this.numRows-1);
        ParameterChecks.assertGreaterEq(rowIndex2, this.numRows-1);

        double temp;
        int row1Start = rowIndex1*numCols;
        int row2Start = rowIndex2*numCols;
        int stop = row1Start + numCols;

        while(row1Start < stop) {
            temp = entries[row1Start];
            entries[row1Start++] = entries[row2Start];
            entries[row2Start++] = temp;
        }

        return this;
    }


    /**
     * Swaps specified columns in the matrix. This is done in place.
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     * @return A reference to this matrix.
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public Matrix swapCols(int colIndex1, int colIndex2) {
        ParameterChecks.assertGreaterEq(0, colIndex1, colIndex2);
        ParameterChecks.assertGreaterEq(colIndex1, this.numCols-1);
        ParameterChecks.assertGreaterEq(colIndex2, this.numCols-1);

        double temp;
        for(int i=0; i<numRows; i++) {
            // Swap elements.
            temp = entries[i*numCols + colIndex1];
            entries[i*numCols + colIndex1] = entries[i*numCols + colIndex2];
            entries[i*numCols + colIndex2] = temp;
        }

        return this;
    }


    @Override
    public Double get(int... indices) {
        ParameterChecks.assertValidIndex(shape, indices);
        return entries[indices[0]*numCols + indices[1]];
    }


    /**
     * Computes the rank of this matrix (i.e. the number of linearly independent rows/columns in this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank (i.e. number of indices needed to specify an entry in
     * the tensor).
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        // Compute the (numerical) matrix rank using the singular value decomposition.
        return new RealSVD(false).decompose(this).getRank();
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     * @return True if this matrix is symmetric. Otherwise, returns false.
     * @see #isAntiSymmetric()
     */
    @Override
    public boolean isSymmetric() {
        return RealDenseProperties.isSymmetric(entries, shape);
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     * @see #isSymmetric()
     */
    @Override
    public boolean isAntiSymmetric() {
        return RealDenseProperties.isAntiSymmetric(entries, shape);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        if(numRows == numCols) {
            return RealDenseProperties.isCloseToIdentity(this.mult(this.T()));
        } else {
            return false;
        }
    }


    /**
     * Computes the complex element-wise square root of a tensor. That is, ff this tensor contains negative values, the
     * resulting root will be complex rather then {@link Double#NaN}.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public CMatrix sqrtComplex() {
        return new CMatrix(shape, ComplexOperations.sqrt(entries));
    }


    /**
     * Gets row of matrix formatted as a human-readable String. Helper method for {@link #toString} method.
     * @param i Index of row to get.
     * @param colStopIndex Stopping index for printing columns.
     * @param maxList List of maximum string representation lengths for each column of this matrix. This
     *                is used to align columns when printing.
     * @return A human-readable String representation of the specified row.
     */
    private String rowToString(int i, int colStopIndex, List<Integer> maxList) {
        int width;
        String value;
        StringBuilder result = new StringBuilder();

        if(i>0) {
            result.append(" [");
        }  else {
            result.append("[");
        }

        for(int j=0; j<colStopIndex; j++) {
            value = StringUtils.ValueOfRound(this.get(i, j), PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + maxList.get(j);
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(PrintOptions.getMaxColumns() < this.numCols) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry in the column now
        value = StringUtils.ValueOfRound(this.get(i, this.numCols-1), PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + maxList.get(maxList.size()-1);
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s]", value));

        return result.toString();
    }


    /**
     * Formats matrix contents as a human-readable String.
     * @return Matrix represented as a human-readable String
     */
    public String toString() {
        StringBuilder result  = new StringBuilder();

        if(PrintOptions.getMaxRows() < this.numRows || PrintOptions.getMaxColumns() < this.numCols) {
            // Then also get the full size of the matrix.
            result.append(String.format("Full Shape: %s\n", this.shape));
        }

        result.append("[");

        if(entries.length==0) {
            result.append("[]"); // No entries in this matrix.
        } else {
            int rowStopIndex = Math.min(PrintOptions.getMaxRows() - 1, this.numRows - 1);
            int colStopIndex = Math.min(PrintOptions.getMaxColumns() - 1, this.numCols - 1);
            int width;
            int totalRowLength = 0; // Total string length of each row (not including brackets)
            String value;

            // Find maximum entry string width in each column so columns can be aligned.
            List<Integer> maxList = new ArrayList<>(colStopIndex + 1);
            for (int j = 0; j < colStopIndex; j++) {
                maxList.add(CNumberUtils.maxStringLength(this.getCol(j).entries, rowStopIndex));
                totalRowLength += maxList.get(maxList.size() - 1);
            }

            if (colStopIndex < this.numCols) {
                maxList.add(CNumberUtils.maxStringLength(this.getCol(this.numCols - 1).entries));
                totalRowLength += maxList.get(maxList.size() - 1);
            }

            if (colStopIndex < this.numCols - 1) {
                totalRowLength += 3 + PrintOptions.getPadding(); // Account for '...' element with padding in each column.
            }

            totalRowLength += maxList.size() * PrintOptions.getPadding(); // Account for column padding

            // Get each row as a string.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(i, colStopIndex, maxList));
                result.append("\n");
            }

            if (PrintOptions.getMaxRows() < this.numRows) {
                width = totalRowLength;
                value = "...";
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format(" [%-" + width + "s]\n", value));
            }

            // Get Last row as a string.
            result.append(rowToString(this.numRows - 1, colStopIndex, maxList));
        }

        result.append("]");

        return result.toString();
    }
}