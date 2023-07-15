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
import com.flag4j.core.MatrixMixin;
import com.flag4j.core.RealMatrixMixin;
import com.flag4j.core.sparse.RealSparseTensorBase;
import com.flag4j.io.PrintOptions;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseMatrixMultiplication;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseMatrixOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseMatrixMultiplication;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseMatrixOperations;
import com.flag4j.operations.sparse.real.*;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseEquals;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseMatrixMultiplication;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseMatrixOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ParameterChecks;
import com.flag4j.util.SparseDataWrapper;
import com.flag4j.util.StringUtils;

import java.util.Arrays;

/**
 * Real sparse matrix. Matrix is stored in coordinate list (COO) format.
 */
public class SparseMatrix
        extends RealSparseTensorBase<SparseMatrix, Matrix, SparseCMatrix, CMatrix>
        implements MatrixMixin<SparseMatrix, Matrix, SparseMatrix, SparseCMatrix, Double, SparseVector, Vector>,
        RealMatrixMixin<SparseMatrix, SparseCMatrix>
{


    /**
     * Row indices of the non-zero entries of the sparse matrix.
     */
    public final int[] rowIndices;
    /**
     * Column indices of the non-zero entries of the sparse matrix.
     */
    public final int[] colIndices;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Creates a square sparse matrix of specified size filled with zeros.
     * @param size The number of rows/columns in this sparse matrix.
     */
    public SparseMatrix(int size) {
        super(new Shape(size, size), 0, new double[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix of specified number of rows and columns filled with zeros.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     */
    public SparseMatrix(int rows, int cols) {
        super(new Shape(rows, cols), 0, new double[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix of specified shape filled with zeros.
     * @param shape Shape of this sparse matrix.
     */
    public SparseMatrix(Shape shape) {
        super(shape, 0, new double[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int size, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices})
        );
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int rows, int cols, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices})
        );

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(Shape shape, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices})
        );
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int size, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices})
        );

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int rows, int cols, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices})
        );
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseMatrix(Shape shape, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                RealDenseTranspose.blockedIntMatrix(new int[][]{rowIndices, colIndices})
        );
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Constructs a sparse tensor whose shape and values are given by another sparse tensor. This effectively copies
     * the tensor.
     * @param A Sparse Matrix to copy.
     */
    public SparseMatrix(SparseMatrix A) {
        super(A.shape.copy(),
                A.nonZeroEntries(),
                A.entries.clone(),
                new int[A.indices.length][A.indices[0].length]
        );
        ArrayUtils.deepCopy(A.indices, this.indices);
        this.rowIndices = A.rowIndices.clone();
        this.colIndices = A.colIndices.clone();
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Checks if an object is equal to this sparse matrix.
     * @param object Object to compare this sparse matrix to.
     * @return True if the object is a matrix (real or complex, dense or sparse) and is element-wise equal to this
     * matrix.
     */
    @Override
    public boolean equals(Object object) {
        boolean equal;

        if(object instanceof Matrix) {
            Matrix mat = (Matrix) object;
            equal = RealDenseSparseEquals.matrixEquals(mat, this);

        } else if(object instanceof CMatrix) {
            CMatrix mat = (CMatrix) object;
            equal = RealComplexDenseSparseEquals.matrixEquals(mat, this);

        } else if(object instanceof SparseMatrix) {
            SparseMatrix mat = (SparseMatrix) object;
            equal = RealSparseEquals.matrixEquals(this, mat);

        } else if(object instanceof SparseCMatrix) {
            SparseCMatrix mat = (SparseCMatrix) object;
            equal = RealComplexSparseEquals.matrixEquals(this, mat);

        } else {
            equal = false;
        }

        return equal;
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
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public SparseMatrix add(SparseMatrix B) {
        return RealSparseMatrixOperations.add(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix add(double a) {
        return RealSparseMatrixOperations.add(this, a);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        return RealComplexSparseMatrixOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public SparseMatrix sub(SparseMatrix B) {
        return RealSparseMatrixOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix sub(double a) {
        return RealSparseMatrixOperations.sub(this, a);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        return RealComplexSparseMatrixOperations.sub(this, a);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseMatrix transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseMatrix T() {
        SparseMatrix transpose = new SparseMatrix(
                shape.copy().swapAxes(0, 1),
                entries.clone(),
                colIndices.clone(),
                rowIndices.clone()
        );

        transpose.sparseSort(); // Ensure the indices are sorted correctly.

        return transpose;
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.assertEquals(indices.length, 2);
        ParameterChecks.assertInRange(indices[0], 0, numRows, "index");
        ParameterChecks.assertInRange(indices[1], 0, numCols, "index");

        return RealSparseElementSearch.matrixGet(this, indices[0], indices[1]);
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public SparseMatrix copy() {
        return new SparseMatrix(shape.copy(), entries.clone(), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public SparseMatrix elemMult(SparseMatrix B) {
        return RealSparseMatrixOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public SparseMatrix elemDiv(Matrix B) {
        return RealDenseSparseMatrixOperations.elemDiv(this, B);
    }


    /**
     * A factory for creating a real sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected SparseMatrix makeTensor(Shape shape, double[] entries, int[][] indices) {
        int[][] rowColIndices = RealDenseTranspose.blockedIntMatrix(indices);
        return new SparseMatrix(shape, entries, rowColIndices[0], rowColIndices[1]);
    }


    /**
     * A factory for creating a real dense tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected Matrix makeDenseTensor(Shape shape, double[] entries) {
        return new Matrix(shape, entries);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected SparseCMatrix makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        int[][] rowColIndices = RealDenseTranspose.blockedIntMatrix(indices);
        return new SparseCMatrix(shape, entries, rowColIndices[0], rowColIndices[1]);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sparseSort() {
        SparseDataWrapper.wrap(entries, rowIndices, colIndices).sparseSort().unwrap(entries, rowIndices, colIndices);
    }



    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public Matrix toDense() {
        double[] entries = new double[totalEntries().intValueExact()];
        int row;
        int col;

        for(int i=0; i<nonZeroEntries; i++) {
            row = rowIndices[i];
            col = colIndices[i];

            entries[row*numCols + col] = this.entries[i];
        }

        return new Matrix(shape.copy(), entries);
    }


    /**
     * Checks if this matrix is the identity matrix.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        return RealSparseMatrixProperties.isIdentity(this);
    }


    /**
     * Checks if matrices are inverses of each other.
     *
     * @param B Second matrix.
     * @return True if matrix B is an inverse of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    @Override
    public boolean isInv(SparseMatrix B) {
        boolean result;

        if(!this.isSquare() || !B.isSquare() || !shape.equals(B.shape)) {
            result = false;
        } else {
            result = this.mult(B).round().isI();
        }

        return result;
    }


    /**
     * Sets an index of this matrix to the specified value. For sparse matrices, a new copy of this matrix
     * with the specified value set will be returned.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A copy of this matrix with the specified value set.
     */
    @Override
    public SparseMatrix set(double value, int row, int col) {
        return RealSparseElementSearch.matrixSet(this, row, col, value);
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
    public SparseMatrix set(Double value, int row, int col) {
        return RealSparseElementSearch.matrixSet(this, row, col, value);
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
    public SparseMatrix setCol(Double[] values, int colIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setCol(Integer[] values, int colIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setCol(double[] values, int colIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setCol(int[] values, int colIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setRow(Double[] values, int rowIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setRow(Integer[] values, int rowIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setRow(double[] values, int rowIndex) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setRow(int[] values, int rowIndex) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseMatrix setSlice(SparseMatrix values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseMatrix setSlice(Double[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseMatrix setSlice(Integer[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseMatrix setSlice(double[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseMatrix setSlice(int[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setSliceCopy(SparseMatrix values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public SparseMatrix removeRow(int rowIndex) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix.
     * @return A copy of this matrix with the specified rows removed.
     */
    @Override
    public SparseMatrix removeRows(int... rowIndices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public SparseMatrix removeCol(int colIndex) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return A copy of this matrix with the specified columns removed.
     */
    @Override
    public SparseMatrix removeCols(int... colIndices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Swaps rows in the matrix.
     *
     * @param rowIndex1 Index of first row to swap.
     * @param rowIndex2 index of second row to swap.
     * @return A reference to this matrix.
     */
    @Override
    public SparseMatrix swapRows(int rowIndex1, int rowIndex2) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Swaps columns in the matrix.
     *
     * @param colIndex1 Index of first column to swap.
     * @param colIndex2 index of second column to swap.
     * @return A reference to this matrix.
     */
    @Override
    public SparseMatrix swapCols(int colIndex1, int colIndex2) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setSliceCopy(Double[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setSliceCopy(Integer[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setSliceCopy(double[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setSliceCopy(int[][] values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
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
    public SparseMatrix setSliceCopy(Matrix values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseMatrix setSlice(Matrix values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
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
        return RealDenseSparseMatrixOperations.add(B, this);
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
        return RealComplexDenseSparseMatrixOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public SparseCMatrix add(SparseCMatrix B) {
        return RealComplexSparseMatrixOperations.add(B, this);
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
        return RealComplexDenseSparseMatrixOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public SparseCMatrix sub(SparseCMatrix B) {
        return RealComplexSparseMatrixOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition of a matrix with a real sparse matrix. The result is stored in this matrix.
     *
     * @param B The sparse matrix to add to this matrix.
     */
    @Override
    public void addEq(SparseMatrix B) {
        // TODO: Implementation.
    }


    /**
     * Computes the element-wise subtraction of this matrix with a real sparse matrix. The result is stored in this matrix.
     *
     * @param B The sparse matrix to subtract from this matrix.
     */
    @Override
    public void subEq(SparseMatrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
        return null;
    }


    @Override
    public Vector mult(SparseVector B) {
        // TODO: Implementation.
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
    public CVector mult(CVector b) {
        // TODO: Implementation.
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
    public CVector mult(SparseCVector b) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(Matrix B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(SparseMatrix B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {{@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CMatrix B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(SparseCMatrix B) {
        // TODO: Implementation.
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
    public Matrix pow(int exponent) {
        // TODO: Implementation.
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
    public SparseMatrix elemMult(Matrix B) {
        return RealDenseSparseMatrixOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public SparseCMatrix elemMult(CMatrix B) {
        return RealComplexDenseSparseMatrixOperations.elemMult(B, this);
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
        return RealComplexSparseMatrixOperations.elemMult(B, this);
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
    public SparseCMatrix elemDiv(CMatrix B) {
        return RealComplexDenseSparseMatrixOperations.elemDiv(this, B);
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double det() {
        // TODO: Implementation. Will need to implement a sparse LU decomposition.
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
        // TODO: Implementation.
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
        // TODO: Implementation.
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
        // TODO: Implementation.
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseMatrix directSum(Matrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix directSum(CMatrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseMatrix invDirectSum(Matrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix invDirectSum(CMatrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    @Override
    public SparseMatrix sumCols() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    @Override
    public SparseMatrix sumRows() {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseMatrix addToEachCol(SparseVector b) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix addToEachCol(SparseCVector b) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseMatrix addToEachRow(SparseVector b) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix addToEachRow(SparseCVector b) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseMatrix stack(SparseMatrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix stack(SparseCMatrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseMatrix stack(SparseMatrix B, int axis) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix stack(SparseCMatrix B, int axis) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseMatrix augment(SparseMatrix B) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix augment(SparseCMatrix B) {
        // TODO: Implementation.
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
    public SparseMatrix stack(Vector b) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix stack(CVector b) {
        // TODO: Implementation.
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
    public SparseCMatrix stack(SparseCVector b) {
        // TODO: Implementation.
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
    public SparseMatrix stack(Vector b, int axis) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix stack(CVector b, int axis) {
        // TODO: Implementation.
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
    public SparseCMatrix stack(SparseCVector b, int axis) {
        // TODO: Implementation.
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
    public SparseMatrix augment(Vector b) {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
    public SparseCMatrix augment(CVector b) {
        // TODO: Implementation.
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
    public SparseCMatrix augment(SparseCVector b) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    @Override
    public SparseMatrix getRow(int i) {
        // TODO: Implementation. Change to return a vector instead of a matrix.
        return null;
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public SparseMatrix getCol(int j) {
        // TODO: Implementation. Change to return a vector instead of a matrix.
        return null;
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param colIdx   Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd   Ending row of the column (exclusive).
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     * @throws IllegalArgumentException   If {@code rowStart} is less than 0.
     * @throws NegativeArraySizeException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public SparseVector getCol(int colIdx, int rowStart, int rowEnd) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public SparseVector toVector() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd   Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd   Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public SparseMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Get a specified column of this matrix at and below a specified row.
     *
     * @param rowStart Index of the row to begin at.
     * @param j        Index of column to get.
     * @return The specified column of this matrix beginning at the specified row.
     * @throws NegativeArraySizeException     If {@code rowStart} is larger than the number of rows in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowStart} or {@code j} is outside the bounds of this matrix.
     */
    @Override
    public SparseMatrix getColBelow(int rowStart, int j) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Get a specified row of this matrix at and after a specified column.
     *
     * @param colStart Index of the row to begin at.
     * @param i        Index of the row to get.
     * @return The specified row of this matrix beginning at the specified column.
     * @throws NegativeArraySizeException     If {@code colStart} is larger than the number of columns in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code i} or {@code colStart} is outside the bounds of this matrix.
     */
    @Override
    public SparseMatrix getRowAfter(int colStart, int i) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets a column of this matrix.
     *
     * @param values Vector containing the new values for the matrix.
     * @param j      Index of the column of this matrix to set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the number of entries in the {@code values} vector
     *                                   is not the same as the number of rows in this matrix.
     * @throws IndexOutOfBoundsException If {@code j} is not within the bounds of this matrix.
     */
    @Override
    public SparseMatrix setCol(SparseVector values, int j) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #tr()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double trace() {
        // TODO: Implementation.
        return null;
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the inverse of this matrix.
     *
     * @return The inverse of this matrix.
     */
    @Override
    public SparseMatrix inv() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the pseudo-inverse of this matrix.
     *
     * @return The pseudo-inverse of this matrix.
     */
    @Override
    public SparseMatrix pInv() {
        // TODO: Implementation.
        return null;
    }
    

    /**
     * Computes the condition number of this matrix using {@link com.flag4j.linalg.decompositions.SVD SVD}.
     * Specifically, the condition number is computed as the maximum singular value divided by the minimum singular
     * value of this matrix.
     *
     * @return The condition number of this matrix (Assuming Frobenius norm).
     */
    @Override
    public double cond() {
        // TODO: Implementation.
        return 0;
    }
    

    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public SparseVector getDiag() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Compute the hermation transpose of this matrix. That is, the complex conjugate transpose of this matrix.
     *
     * @return The complex conjugate transpose of this matrix.
     */
    @Override
    public SparseMatrix H() {
        return T();
    }


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * rows in matrix {@code B}.
     */
    @Override
    public Matrix mult(SparseMatrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new Matrix(numRows, B.numCols,
                RealSparseMatrixMultiplication.concurrentStandard(
                    entries, rowIndices, colIndices, shape,
                    B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );
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
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new CMatrix(numRows, B.numCols,
                RealComplexDenseSparseMatrixMultiplication.concurrentStandard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.shape
                )
        );
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
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new CMatrix(numRows, B.numCols,
                RealComplexSparseMatrixMultiplication.concurrentStandard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );
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
        ParameterChecks.assertMatMultShapes(shape, new Shape(b.size, 1));

        return new Vector(
                RealDenseSparseMatrixMultiplication.concurrentStandardVector(
                        this.entries, this.rowIndices, this.colIndices, this.shape,
                        b.entries, b.shape
                )
        );
    }


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    @Override
    public boolean isSquare() {
        return shape.dims[0]==shape.dims[1];
    }


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     *
     * @return True if this matrix can be represented as either a row or column vector.
     */
    @Override
    public boolean isVector() {
        // TODO: Implementation.
        return false;
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
        // TODO: Implementation.
        return 0;
    }


    /**
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     *
     * @return True is this matrix is triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTri() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriL() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriU() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if this matrix is diagonal.
     *
     * @return True is this matrix is diagonal. Otherwise, returns false.
     */
    @Override
    public boolean isDiag() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        // TODO: Implementation.
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
        // TODO: Implementation.
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
        // TODO: Implementation.
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
        // TODO: Implementation.
        return 0;
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @return The max norm of this matrix.
     */
    @Override
    public double maxNorm() {
        // TODO: Implementation.
        return 0;
    }


    /**
     * Computes the rank of this matrix (i.e. the dimension of the column space of this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank.
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        // TODO: Implementation.
        return 0;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isAntiSymmetric() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Computes the complex element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public SparseCMatrix sqrtComplex() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public SparseCMatrix toComplex() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected SparseMatrix getSelf() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public SparseMatrix set(double value, int... indices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public SparseMatrix flatten(int axis) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        // TODO: Implementation.
        return 0;
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
        // TODO: Implementation.
        return 0;
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nonZeroEntries;
        StringBuilder result = new StringBuilder(String.format("Full Shape: %s\n", shape));
        result.append("Non-zero entries: [");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry now
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value)).append("]\n");

        result.append("Row Indices: ").append(Arrays.toString(rowIndices)).append("\n");
        result.append("Col Indices: ").append(Arrays.toString(colIndices));

        return result.toString();
    }
}
