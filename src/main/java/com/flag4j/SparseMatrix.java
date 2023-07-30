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
import com.flag4j.operations.TransposeDispatcher;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.dense.real.RealDenseOperations;
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
import com.flag4j.util.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

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
     * Creates a sparse matrix with specified shape, non-zero entries, and indices.
     * @param shape Shape of the sparse matrix.
     * @param entries Non-zero entries of the sparse matrix.
     * @param rowIndices Non-zero row indices of the sparse matrix.
     * @param colIndices Non-zero column indices of the sparse matrix.
     */
    public SparseMatrix(Shape shape, List<Double> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(
            shape,
            entries.size(),
            ArrayUtils.fromDoubleList(entries),
            new int[rowIndices.size()][2]
        );
        this.rowIndices = ArrayUtils.fromIntegerList(rowIndices);
        this.colIndices = ArrayUtils.fromIntegerList(colIndices);

        int[][] indices = RealDenseTranspose.blockedIntMatrix(new int[][]{this.rowIndices, this.colIndices});
        ArrayUtils.deepCopy(indices, this.indices);

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
        ParameterChecks.assertIndexInBounds(numRows, indices[0]);
        ParameterChecks.assertIndexInBounds(numCols, indices[1]);

        return RealSparseMatrixGetSet.matrixGet(this, indices[0], indices[1]);
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
        ParameterChecks.assertIndexInBounds(numRows, row);
        ParameterChecks.assertIndexInBounds(numCols, col);

        return RealSparseMatrixGetSet.matrixSet(this, row, col, value);
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
        return RealSparseMatrixGetSet.matrixSet(this, row, col, value);
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
        return RealSparseMatrixGetSet.setRow(
                this,
                colIndex,
                Stream.of(values).mapToDouble(Double::doubleValue).toArray()
        );
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
        return RealSparseMatrixGetSet.setCol(
                this,
                colIndex,
                Stream.of(values).mapToDouble(Integer::doubleValue).toArray()
        );
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
        return RealSparseMatrixGetSet.setCol(this, colIndex, values);
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
        return RealSparseMatrixGetSet.setRow(
                this,
                colIndex,
                Arrays.stream(values).asDoubleStream().toArray()
        );
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
        return RealSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                Stream.of(values).mapToDouble(Double::doubleValue).toArray()
        );
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A copy of this matrix with the specified row set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public SparseMatrix setRow(Integer[] values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                Stream.of(values).mapToDouble(Integer::doubleValue).toArray()
        );
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A copy of this matrix with the specified row set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public SparseMatrix setRow(double[] values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(this, rowIndex, values);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A copy of this matrix with the specified row set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public SparseMatrix setRow(int[] values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                Arrays.stream(values).asDoubleStream().toArray()
        );
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
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public SparseMatrix removeRow(int rowIndex) {
        return RealSparseMatrixManipulations.removeRow(this, rowIndex);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. The indices must be sorted and unique
     *                   otherwise the behavior of this method is undefined.
     *                   {@link Arrays#sort(int[]) Arrays.sort(rowIndices)} must be called first if the array is not
     *                   already sorted.
     * @return A copy of this matrix with the specified rows removed.
     */
    @Override
    public SparseMatrix removeRows(int... rowIndices) {
        return RealSparseMatrixManipulations.removeRows(this, rowIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public SparseMatrix removeCol(int colIndex) {
        return RealSparseMatrixManipulations.removeCol(this, colIndex);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return A copy of this matrix with the specified columns removed.
     */
    @Override
    public SparseMatrix removeCols(int... colIndices) {
        return RealSparseMatrixManipulations.removeCols(this, colIndices);
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
        return RealSparseMatrixManipulations.swapRows(this, rowIndex1, rowIndex2);
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
        return RealSparseMatrixManipulations.swapCols(this, colIndex1, colIndex2);
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
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(Matrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);
        double[] dest = RealDenseSparseMatrixMultiplication.concurrentStandard(
                entries, rowIndices, colIndices, shape,
                B.entries, B.shape
        );
        return new Matrix(new Shape(numRows, B.numCols), dest);
    }


    @Override
    public Vector mult(SparseVector B) {
        double[] dest = RealSparseMatrixMultiplication.concurrentStandardVector(
                entries, rowIndices, colIndices, shape,
                B.entries, B.indices
        );
        return new Vector(dest);
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
        CNumber[] dest = RealComplexDenseSparseMatrixMultiplication.concurrentStandardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.shape
        );
        return new CVector(dest);
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
        CNumber[] dest = RealComplexSparseMatrixMultiplication.concurrentStandardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices, b.shape
        );
        return new CVector(dest);
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
        Matrix product = new Matrix(
                shape.copy(),
                RealDenseSparseMatrixMultiplication.concurrentStandard(
                    entries, rowIndices, colIndices, shape,
                    B.entries, B.shape
                )
        );

        return TransposeDispatcher.dispatch(product);
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
        Matrix product = new Matrix(
                shape.copy(),
                RealSparseMatrixMultiplication.concurrentStandard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );

        return TransposeDispatcher.dispatch(product);
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
        CMatrix product = new CMatrix(
                shape.copy(),
                RealComplexDenseSparseMatrixMultiplication.concurrentStandard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.shape
                )
        );

        return TransposeDispatcher.dispatch(product);
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
        CMatrix product = new CMatrix(
                shape.copy(),
                RealComplexSparseMatrixMultiplication.concurrentStandard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );

        return TransposeDispatcher.dispatch(product);
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method will be significantly
     * faster.
     *
     * @param exponent The exponent in the matrix power. If {@code exponent = 0} then the identity matrix will be
     *                 returned.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     * @throws IllegalArgumentException If {@code exponent} is negative.
     * @throws IllegalArgumentException If this sparse matrix is not square.
     */
    @Override
    public Matrix pow(int exponent) {
        ParameterChecks.assertSquare(shape);
        ParameterChecks.assertGreaterEq(0, exponent);

        Matrix power;

        if(exponent==0) {
            power = Matrix.I(numRows);
        } else if(exponent==1) {
            power = this.toDense();
        } else {
            // Compute the first sparse-sparse matrix multiplication.
            double[] destEntries = RealSparseMatrixMultiplication.concurrentStandard(
                    entries, rowIndices, colIndices, shape,
                    entries, rowIndices, colIndices, shape
            );

            // Compute the remaining dense-sparse matrix multiplication.
            for(int i=2; i<exponent; i++) {
                destEntries = RealDenseSparseMatrixMultiplication.concurrentStandard(
                        destEntries, shape,
                        entries, rowIndices, colIndices, shape
                );
            }

            power = new Matrix(shape.copy(), destEntries);
        }

        return power;
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
     * <p><b>WARNING:</b> Currently, this method will convert this matrix to a dense matrix.</p>
     *
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double det() {
        return toDense().det();
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
    public Double fib(SparseMatrix B) {
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
    public CNumber fib(SparseCMatrix B) {
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
    public SparseMatrix directSum(Matrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+entries.length;
            indices = B.shape.getIndices(i);
            destRowIndices[destIdx] = indices[0] + numRows;
            destColIndices[destIdx] = indices[1] + numCols;
        }

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseMatrix directSum(SparseMatrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+entries.length;
            destRowIndices[destIdx] = B.rowIndices[i] + numRows;
            destColIndices[destIdx] = B.colIndices[i] + numCols;
        }

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix directSum(CMatrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+entries.length;
            indices = B.shape.getIndices(i);
            destRowIndices[destIdx] = indices[0] + numRows;
            destColIndices[destIdx] = indices[1] + numCols;
        }

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix directSum(SparseCMatrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+entries.length;
            destRowIndices[destIdx] = B.rowIndices[i] + numRows;
            destColIndices[destIdx] = B.colIndices[i] + numCols;
        }

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseMatrix invDirectSum(Matrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = colIndices.clone();
        ArrayUtils.shift(B.numCols, shiftedColIndices);
        int[] shiftedRowIndices = ArrayUtils.rangeInt(numRows, destRowIndices.length);
        int[] bColIndices = ArrayUtils.rangeInt(0, B.numCols);

        // Copy shifted indices of both matrices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, shiftedColIndices.length, bColIndices.length);

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseMatrix invDirectSum(SparseMatrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = colIndices.clone();
        ArrayUtils.shift(B.numCols, shiftedColIndices);
        int[] shiftedRowIndices = B.rowIndices.clone();
        ArrayUtils.shift(numRows, shiftedRowIndices);

        // Copy shifted indices of both matrices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, shiftedColIndices.length, B.colIndices.length);

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix invDirectSum(CMatrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = colIndices.clone();
        ArrayUtils.shift(B.numCols, shiftedColIndices);
        int[] shiftedRowIndices = ArrayUtils.rangeInt(numRows, destRowIndices.length);
        int[] bColIndices = ArrayUtils.rangeInt(0, B.numCols);

        // Copy shifted indices of both matrices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, shiftedColIndices.length, bColIndices.length);

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix invDirectSum(SparseCMatrix B) {
        Shape destShape = new Shape(numRows + B.numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = colIndices.clone();
        ArrayUtils.shift(B.numCols, shiftedColIndices);
        int[] shiftedRowIndices = B.rowIndices.clone();
        ArrayUtils.shift(numRows, shiftedRowIndices);

        // Copy shifted indices of both matrices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, shiftedColIndices.length, B.colIndices.length);

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    @Override
    public SparseMatrix sumCols() {
        List<Double> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<this.entries.length; i++) {
            int idx = rowIndices.indexOf(this.rowIndices[i]);

            if(idx < 0) {
                // No value with this row index exists.
                entries.add(this.entries[i]);
                rowIndices.add(this.rowIndices[i]);
                colIndices.add(0);
            } else {
                // A value already exists with this row index. Update it.
                entries.set(idx, entries.get(i) + this.entries[i]);
            }
        }

        return new SparseMatrix(new Shape(this.numRows, 1), entries, rowIndices, colIndices);
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    @Override
    public SparseMatrix sumRows() {
        List<Double> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<this.entries.length; i++) {
            int idx = rowIndices.indexOf(this.colIndices[i]);

            if(idx < 0) {
                // No value with this column index exists.
                entries.add(this.entries[i]);
                rowIndices.add(0);
                colIndices.add(this.colIndices[i]);
            } else {
                // A value already exists with this column index. Update it.
                entries.set(idx, entries.get(i) + this.entries[i]);
            }
        }

        return new SparseMatrix(new Shape(1, numCols), entries, rowIndices, colIndices);
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
        return RealDenseSparseMatrixOperations.addToEachCol(this, b);
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
        return RealSparseMatrixOperations.addToEachCol(this, b);
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
        return RealComplexDenseSparseMatrixOperations.addToEachCol(this, b);
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
        return RealComplexSparseMatrixOperations.addToEachCol(this, b);
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
        return RealDenseSparseMatrixOperations.addToEachRow(this, b);
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
        return RealSparseMatrixOperations.addToEachRow(this, b);
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
        return RealComplexDenseSparseMatrixOperations.addToEachRow(this, b);
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
        return RealComplexSparseMatrixOperations.addToEachRow(this, b);
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
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numCols, numRows+B.numRows);
        double[] destEntries = new double[destShape.totalEntries().intValueExact()];

        // Copy values from B
        System.arraycopy(B.entries, 0, destEntries, shape.totalEntries().intValueExact(), B.entries.length);

        // Copy non-zero values from this matrix.
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*numCols + colIndices[i]] = entries[i];
        }

        return new Matrix(destShape, destEntries);
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
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numCols, numRows+B.numRows);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(B.rowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, colIndices.length, B.colIndices.length);

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numCols, numRows+B.numRows);
        CNumber[] destEntries = new CNumber[destShape.totalEntries().intValueExact()];

        // Copy values from B
        ArrayUtils.arraycopy(B.entries, 0, destEntries, shape.totalEntries().intValueExact(), B.entries.length);

        // Copy non-zero values from this matrix (and set zero values to zero.).
        ArrayUtils.fillZerosRange(destEntries, 0, shape.totalEntries().intValueExact());
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*numCols + colIndices[i]] = new CNumber(entries[i]);
        }

        return new CMatrix(destShape, destEntries);
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
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numCols, numRows+B.numRows);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(B.rowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, colIndices.length, B.colIndices.length);

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(B) : this.stack(B);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(B) : this.stack(B);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(B) : this.stack(B);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(B) : this.stack(B);
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
        ParameterChecks.assertEquals(numRows, B.numRows);

        Shape destShape = new Shape(numRows, numCols + B.numCols);
        double[] destEntries = new double[destShape.totalEntries().intValueExact()];

        // Copy sparse values.
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*destShape.dims[1] + colIndices[i]] = entries[i];
        }

        // Copy dense values by row.
        for(int i=0; i<numRows; i++) {
            int startIdx = i*destShape.dims[1] + numCols;
            System.arraycopy(B.entries, i*B.numCols, destEntries, startIdx, B.numCols);
        }

        return new Matrix(destShape, destEntries);
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
        ParameterChecks.assertEquals(numRows, B.numRows);

        Shape destShape = new Shape(numRows, numCols + B.numCols);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(B.rowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, colIndices.length, B.colIndices.length);
        
        SparseMatrix dest = new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sparseSort(); // Ensure indices are sorted properly.

        return dest;
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
        ParameterChecks.assertEquals(numRows, B.numRows);

        Shape destShape = new Shape(numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[destShape.totalEntries().intValueExact()];

        // Copy sparse values.
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*destShape.dims[1] + colIndices[i]] = new CNumber(entries[i]);
        }

        // Copy dense values by row.
        for(int i=0; i<numRows; i++) {
            int startIdx = i*destShape.dims[1] + numCols;
            ArrayUtils.arraycopy(B.entries, i*B.numCols, destEntries, startIdx, B.numCols);
        }

        return new CMatrix(destShape, destEntries);
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
        ParameterChecks.assertEquals(numRows, B.numRows);

        Shape destShape = new Shape(numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(B.rowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, colIndices.length, B.colIndices.length);

        SparseCMatrix dest = new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sparseSort(); // Ensure indices are sorted properly.

        return dest;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the vector will be treated as a row vector.<br>
     * Also see {@link #stack(Vector, int)} and {@link #augment(Vector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public SparseMatrix stack(Vector b) {
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        double[] destEntries = new double[entries.length + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy values from vector and create indices.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.size);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(ArrayUtils.rangeInt(0, numCols), 0, destColIndices, entries.length, numCols);

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        double[] destEntries = new double[entries.length + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy values and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(b.indices, 0, destColIndices, entries.length, b.entries.length);

        return new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        CNumber[] destEntries = new CNumber[entries.length + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);

        // Copy values from vector and create indices.
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.size);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(ArrayUtils.rangeInt(0, numCols), 0, destColIndices, entries.length, numCols);

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        CNumber[] destEntries = new CNumber[entries.length + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy values and indices from vector.
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(b.indices, 0, destColIndices, entries.length, b.entries.length);

        return new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(b) : this.stack(b);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(b) : this.stack(b);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(b) : this.stack(b);
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
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        return axis==0 ? this.augment(b) : this.stack(b);
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix.<br>
     * Also see {@link #stack(Vector)} and {@link #stack(Vector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public SparseMatrix augment(Vector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        double[] destEntries = new double[nonZeroEntries + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(ArrayUtils.rangeInt(0, numRows), 0, destRowIndices, entries.length, numRows);
        Arrays.fill(destColIndices, entries.length, numRows, numCols);

        SparseMatrix dest = new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sparseSort(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix.<br>
     * Also see {@link #stack(SparseVector)} and {@link #stack(SparseVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public SparseMatrix augment(SparseVector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        double[] destEntries = new double[nonZeroEntries + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(b.indices, 0, destRowIndices, entries.length, b.entries.length);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        SparseMatrix dest = new SparseMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sparseSort(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. <br>
     * Also see {@link #stack(CVector)} and {@link #stack(CVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public SparseCMatrix augment(CVector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        CNumber[] destEntries = new CNumber[nonZeroEntries + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(ArrayUtils.rangeInt(0, numRows), 0, destRowIndices, entries.length, numRows);
        Arrays.fill(destColIndices, entries.length, numRows, numCols);

        SparseCMatrix dest = new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sparseSort(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix.<br>
     * Also see {@link #stack(SparseCVector)} and {@link #stack(SparseCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public SparseCMatrix augment(SparseCVector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        CNumber[] destEntries = new CNumber[nonZeroEntries + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(b.indices, 0, destRowIndices, entries.length, b.entries.length);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        SparseCMatrix dest = new SparseCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sparseSort(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    @Override
    public SparseMatrix getRow(int i) {
        // TODO: Change to return a vector instead of a matrix.
        return RealSparseMatrixGetSet.getRow(this, i);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public SparseMatrix getCol(int j) {
        // TODO: Change to return a vector instead of a matrix.
        return RealSparseMatrixGetSet.getCol(this, j);
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
        return RealSparseMatrixGetSet.getCol(this, colIdx, rowStart, rowEnd);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public SparseVector toVector() {
        int[] destIndices = new int[indices.length];

        for(int i=0; i<entries.length; i++) {
            destIndices[i] = rowIndices[i]*colIndices[i];
        }

        return new SparseVector(numRows*numCols, entries.clone(), destIndices);
    }


    /**
     * Converts this matrix to an equivalent complex tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public SparseTensor toTensor() {
        int[][] destIndices = new int[indices.length][indices[0].length];
        ArrayUtils.deepCopy(indices, destIndices);

        return new SparseTensor(this.shape.copy(), this.entries.clone(), destIndices);
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
        return RealSparseMatrixGetSet.getSlice(this, rowStart, rowEnd, colStart, colEnd);
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
        // TODO: Change so that a vector is returned.
        return RealSparseMatrixGetSet.getCol(this, j, rowStart, numRows).toMatrix(true);
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
        // TODO: Change so that a sparse vector is returned instead.
        return RealSparseMatrixGetSet.getRow(this, i, colStart, numCols).toMatrix(false);
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
        // TODO: Change so that a sparse vector is returned instead.
        return RealSparseMatrixGetSet.setCol(this, j, values);
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
        return tr();
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
        double trace = 0;

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                trace += entries[i];
            }
        }

        return trace;
    }


    /**
     * Computes the inverse of this matrix.
     *
     * <p><b>WARNING:</b> Currently, this method will convert this matrix to a dense matrix.</p>
     *
     * @return The inverse of this matrix.
     */
    @Override
    public Matrix inv() {
        return toDense().inv();
    }


    /**
     * Computes the pseudo-inverse of this matrix.
     *
     * <p><b>WARNING:</b> Currently, this method will convert this matrix to a dense matrix.</p>
     *
     * @return The pseudo-inverse of this matrix.
     */
    @Override
    public Matrix pInv() {
        return toDense().inv();
    }
    

    /**
     * Computes the condition number of this matrix using {@link com.flag4j.linalg.decompositions.SVD SVD}.
     * Specifically, the condition number is computed as the maximum singular value divided by the minimum singular
     * value of this matrix.
     *
     * <p>
     *     WARNING: This method will convert the sparse matrix to a dense matrix to perform the computation.
     * </p>
     *
     * @return The condition number of this matrix (Assuming Frobenius norm).
     */
    @Override
    public double cond() {
        return toDense().cond();
    }
    

    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public SparseVector getDiag() {
        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                destEntries.add(entries[i]);
                destIndices.add(rowIndices[i]);
            }
        }

        return new SparseVector(
                numRows,
                destEntries.stream().mapToDouble(Double::doubleValue).toArray(),
                destIndices.stream().mapToInt(Integer::intValue).toArray()
        );
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
        return numRows==1 || numCols==1;
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
        int type = -1;

        if(numRows==1 && numCols==1) {
            type=0;
        } else if(numRows==1) {
            type=1;
        } else if(numCols==1) {
            type=2;
        }

        return type;
    }


    /**
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     *
     * @return True is this matrix is triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTri() {
        return isTriL() || isTriU();
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriL() {
        boolean result = true;

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i] < colIndices[i]) {
                // Then entry is not in lower triangle.
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriU() {
        boolean result = true;

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i] > colIndices[i]) {
                // Then entry is not in upper triangle.
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if this matrix is diagonal.
     *
     * @return True is this matrix is diagonal. Otherwise, returns false.
     */
    @Override
    public boolean isDiag() {
        boolean result = true;

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]!=colIndices[i]) {
                // Then entry is not the diagonal.
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * <p>
     *     WARNING: This method will convert this matrix to a dense matrix.
     * </p>
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return toDense().isFullRank();
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.
     *
     * <p><b>WARNING:</b> Currently, this method will convert this matrix to a dense matrix.</p>
     *
     * @return True if this matrix is singular. Otherwise, returns false.
     * @see #isInvertible()
     */
    @Override
    public boolean isSingular() {
        return toDense().isSingular();
    }


    /**
     * Checks if a matrix is invertible.
     *
     * @return True if this matrix is invertible.
     * @see #isSingular()
     */
    @Override
    public boolean isInvertible() {
        return !isSingular();
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
        // Sparse implementation is only faster for very sparse matrices.
        return sparsity()>=0.95 ? RealSparseNorms.matrixNormLpq(this, p, q) :
                toDense().norm(p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @return The max norm of this matrix.
     */
    @Override
    public double maxNorm() {
        return RealDenseOperations.matrixMaxNorm(entries);
    }


    /**
     * Computes the rank of this matrix (i.e. the dimension of the column space of this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank.
     *
     * <p>
     *     <b>WARNING</b>: This method will convert this matrix to a dense matrix.
     * </p>
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        return toDense().matrixRank();
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        return RealSparseMatrixProperties.isSymmetric(this);
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isAntiSymmetric() {
        return RealSparseMatrixProperties.isAntiSymmetric(this);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        if(isSquare()) {
            return this.mult(this.T()).round().equals(Matrix.I(numRows));
        } else {
            return false;
        }
    }


    /**
     * Computes the complex element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public SparseCMatrix sqrtComplex() {
        return new SparseCMatrix(shape.copy(), ComplexOperations.sqrt(entries), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public SparseCMatrix toComplex() {
        CNumber[] dest = new CNumber[entries.length];
        ArrayUtils.copy2CNumber(entries, dest);
        return new SparseCMatrix(shape.copy(), dest, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected SparseMatrix getSelf() {
        return this;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException If there are not exactly two {@code indices} provided.
     */
    @Override
    public SparseMatrix set(double value, int... indices) {
        ParameterChecks.assertEquals(2, indices.length);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public SparseMatrix flatten(int axis) {
        ParameterChecks.assertIndexInBounds(2, axis);
        int[] dims = {1, 1};
        dims[1-axis] = this.totalEntries().intValueExact();

        int[] rowIndices = new int[this.rowIndices.length];
        int[] colIndices = new int[this.colIndices.length];

        if(axis==0) {
            // Flatten to a single row.
            colIndices = this.colIndices.clone();
        } else {
            // Flatten to a single column.
            rowIndices = this.rowIndices.clone();
        }

        return new SparseMatrix(new Shape(dims), entries.clone(), rowIndices, colIndices);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        // Sparse implementation is only faster for very sparse matrices.
        return sparsity()>=0.95 ? RealSparseNorms.matrixNormL2(this) :
                toDense().norm();
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
        // Sparse implementation is only faster for very sparse matrices.
        return sparsity()>=0.95 ? RealSparseNorms.matrixNormLp(this, p) :
                toDense().norm(p);
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

        if(entries.length > 0) {
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
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Indices: ").append(Arrays.toString(rowIndices)).append("\n");
        result.append("Col Indices: ").append(Arrays.toString(colIndices));

        return result.toString();
    }
}




