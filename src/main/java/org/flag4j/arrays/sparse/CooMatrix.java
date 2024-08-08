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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.MatrixMixin;
import org.flag4j.core.RealMatrixMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.sparse_base.RealSparseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.dense.real.RealDenseTranspose;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseMatrixOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixOperations;
import org.flag4j.operations.sparse.coo.SparseDataWrapper;
import org.flag4j.operations.sparse.coo.real.*;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexSparseMatrixMultiplication;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexSparseMatrixOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>Real sparse matrix. Matrix is stored in coordinate list (COO) format.</p>
 *
 * <p>COO matrices are best suited for efficient modification and construction of sparse matrices. Coo matrices are <b>not</b> well
 * suited for matrix-matrix or matrix-vector multiplication (see {@link CsrMatrix}).</p>
 *
 * <p>If a sparse matrix needs to be incrementally constructed, then a COO matrix should be used to construct the matrix as it
 * allows for efficient modification. If the matrix then needs to be used in a matrix-matrix or matrix-vector multiplication
 * problem, it should first be converted to a {@link CsrMatrix} in most cases.</p>
 *
 * @see CsrMatrix
 * @see CooCMatrix
 */
public class CooMatrix
        extends RealSparseTensorBase<CooMatrix, Matrix, CooCMatrix, CMatrix>
        implements MatrixMixin<CooMatrix, Matrix, CooMatrix, CooCMatrix, CooCMatrix, Double, CooVector, Vector>,
        RealMatrixMixin<CooMatrix, CooCMatrix>
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
    public CooMatrix(int size) {
        super(new Shape(size, size), 0, new double[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse matrix of specified number of rows and columns filled with zeros.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     */
    public CooMatrix(int rows, int cols) {
        super(new Shape(rows, cols), 0, new double[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse matrix of specified shape filled with zeros.
     * @param shape Shape of this sparse matrix.
     */
    public CooMatrix(Shape shape) {
        super(shape, 0, new double[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public CooMatrix(int size, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                nonZeroEntries,
                rowIndices, colIndices
        );

        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public CooMatrix(int rows, int cols, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                nonZeroEntries,
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public CooMatrix(Shape shape, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                nonZeroEntries,
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public CooMatrix(int size, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                ArrayUtils.asDouble(nonZeroEntries, null),
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public CooMatrix(int rows, int cols, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                ArrayUtils.asDouble(nonZeroEntries, null),
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.get(0);
        numCols = shape.get(1);
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
    public CooMatrix(Shape shape, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                ArrayUtils.asDouble(nonZeroEntries, null),
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Constructs a sparse tensor whose shape and values are given by another sparse tensor. This effectively copies
     * the tensor.
     * @param A Sparse Matrix to copy.
     */
    public CooMatrix(CooMatrix A) {
        super(A.shape,
                A.nonZeroEntries(),
                A.entries.clone(),
                A.rowIndices.clone(),
                A.colIndices.clone()
        );
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, and indices.
     * @param shape Shape of the sparse matrix.
     * @param entries Non-zero entries of the sparse matrix.
     * @param rowIndices Non-zero row indices of the sparse matrix.
     * @param colIndices Non-zero column indices of the sparse matrix.
     */
    public CooMatrix(Shape shape, List<Double> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(
            shape,
            entries.size(),
            ArrayUtils.fromDoubleList(entries),
            ArrayUtils.fromIntegerList(rowIndices),
            ArrayUtils.fromIntegerList(colIndices)
        );
        ParameterChecks.assertEquals(entries.size(), rowIndices.size(), colIndices.size());
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Checks if an object is equal to this sparse COO matrix.
     * @param object Object to compare this sparse COO matrix to.
     * @return True if the object is a {@link CooMatrix}, has the same shape as this matrix, and is element-wise equal to this
     * matrix.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooMatrix src2 = (CooMatrix) object;
        return RealSparseEquals.matrixEquals(this, src2);
    }


    /**
     * Constructs a sparse COO matrix from a dense matrix. Any value that is not exactly zero will be considered a non-zero value.
     * @param src Dense matrix to convert to sparse COO matrix.
     * @return An sparse COO matrix equivalent to the dense {@code src} matrix.
     */
    public static CooMatrix fromDense(Matrix src) {
        int rows = src.numRows;
        int cols = src.numCols;
        List<Double> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                double val = src.entries[rowOffset + j];

                if(val != 0d) {
                    entries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooMatrix(src.shape, entries, rowIndices, colIndices);
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
     * Converts this COO matrix to an equivalent CSR matrix.
     * @return An equivalent sparse {@link CsrMatrix CSR Matrix}.
     */
    public CsrMatrix toCsr() {
        return new CsrMatrix(this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooMatrix add(CooMatrix B) {
        return RealSparseMatrixOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CooMatrix add(CsrMatrix B) {
        return add(B.toCoo());
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CooCMatrix add(CsrCMatrix B) {
        return add(B.toCoo());
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
    public CooMatrix sub(CooMatrix B) {
        return RealSparseMatrixOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CooMatrix sub(CsrMatrix B) {
        return sub(B.toCoo());
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CooCMatrix sub(CsrCMatrix B) {
        return sub(B.toCoo());
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
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooMatrix T() {
        CooMatrix transpose = new CooMatrix(
                shape.swapAxes(0, 1),
                entries.clone(),
                colIndices.clone(),
                rowIndices.clone()
        );

        transpose.sortIndices(); // Ensure the indices are sorted correctly.

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
    public CooMatrix copy() {
        return new CooMatrix(shape, entries.clone(), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooMatrix elemMult(CooMatrix B) {
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
    public CooMatrix elemDiv(Matrix B) {
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
    protected CooMatrix makeTensor(Shape shape, double[] entries, int[][] indices) {
        int[][] rowColIndices = RealDenseTranspose.blockedIntMatrix(indices);
        return new CooMatrix(shape, entries, rowColIndices[0], rowColIndices[1]);
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
    protected CooCMatrix makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        int[][] rowColIndices = RealDenseTranspose.blockedIntMatrix(indices);
        return new CooCMatrix(shape, entries, rowColIndices[0], rowColIndices[1]);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
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

        for(int i = 0; i< nnz; i++) {
            row = rowIndices[i];
            col = colIndices[i];

            entries[row*numCols + col] = this.entries[i];
        }

        return new Matrix(shape, entries);
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
     * Sets an index of this matrix to the specified value. For sparse matrices, a new copy of this matrix
     * with the specified value set will be returned.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A copy of this matrix with the specified value set.
     */
    @Override
    public CooMatrix set(double value, int row, int col) {
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
    public CooMatrix set(Double value, int row, int col) {
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
    public CooMatrix setCol(Double[] values, int colIndex) {
        return RealSparseMatrixGetSet.setCol(
                this,
                colIndex,
                ArrayUtils.unbox(values)
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
    public CooMatrix setCol(Integer[] values, int colIndex) {
        return RealSparseMatrixGetSet.setCol(
                this,
                colIndex,
                ArrayUtils.asDouble(values, null)
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
    public CooMatrix setCol(double[] values, int colIndex) {
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
    public CooMatrix setCol(Vector values, int colIndex) {
        return RealSparseMatrixGetSet.setCol(this, colIndex, values.entries);
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
    public CooMatrix setCol(int[] values, int colIndex) {
        return RealSparseMatrixGetSet.setCol(
                this,
                colIndex,
                ArrayUtils.asDouble(values, null)
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
    public CooMatrix setRow(Double[] values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                ArrayUtils.unbox(values)
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
    public CooMatrix setRow(Integer[] values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                ArrayUtils.asDouble(values, null)
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
    public CooMatrix setRow(double[] values, int rowIndex) {
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
    public CooMatrix setRow(int[] values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                ArrayUtils.asDouble(values, null)
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
    public CooMatrix setSlice(CooMatrix values, int rowStart, int colStart) {
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
    public CooMatrix setSlice(Double[][] values, int rowStart, int colStart) {
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
    public CooMatrix setSlice(Integer[][] values, int rowStart, int colStart) {
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
    public CooMatrix setSlice(double[][] values, int rowStart, int colStart) {
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
    public CooMatrix setSlice(int[][] values, int rowStart, int colStart) {
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public CooMatrix removeRow(int rowIndex) {
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
    public CooMatrix removeRows(int... rowIndices) {
        return RealSparseMatrixManipulations.removeRows(this, rowIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public CooMatrix removeCol(int colIndex) {
        return RealSparseMatrixManipulations.removeCol(this, colIndex);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return A copy of this matrix with the specified columns removed.
     */
    @Override
    public CooMatrix removeCols(int... colIndices) {
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
    public CooMatrix swapRows(int rowIndex1, int rowIndex2) {
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
    public CooMatrix swapCols(int colIndex1, int colIndex2) {
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
    public CooMatrix setSlice(Matrix values, int rowStart, int colStart) {
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
    public CooCMatrix add(CooCMatrix B) {
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
    public CooCMatrix sub(CooCMatrix B) {
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
        double[] dest = RealDenseSparseMatrixMultiplication.standard(
                entries, rowIndices, colIndices, shape,
                B.entries, B.shape
        );
        return new Matrix(new Shape(numRows, B.numCols), dest);
    }


    @Override
    public Vector mult(CooVector b) {
        double[] dest = RealSparseMatrixMultiplication.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices
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
        CNumber[] dest = RealComplexDenseSparseMatrixMultiplication.standardVector(
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
    public CVector mult(CooCVector b) {
        CNumber[] dest = RealComplexSparseMatrixMultiplication.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices, b.shape
        );
        return new CVector(dest);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method <i>may</i>
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(Matrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        Matrix Bt = TransposeDispatcher.dispatch(B);

        return new Matrix(
                numRows, Bt.numCols,
                RealDenseSparseMatrixMultiplication.standard(
                    entries, rowIndices, colIndices, shape,
                    Bt.entries, Bt.shape
                )
        );
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method <i>may</i>
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(CooMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        CooMatrix Bt = B.transpose();

        return new Matrix(
                numRows, Bt.numCols,
                RealSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        Bt.entries, Bt.rowIndices, Bt.colIndices, Bt.shape
                )
        );
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {{@code this.mult(B.T())}.
     * For large matrices, this method <i>may</i>
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        CMatrix Bt = TransposeDispatcher.dispatch(B);

        return new CMatrix(
                numRows, Bt.numCols,
                RealComplexDenseSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        Bt.entries, Bt.shape
                )
        );
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method <i>may</i>
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CooCMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        CooCMatrix Bt = B.T();

        return new CMatrix(
                numRows, Bt.numCols,
                RealComplexSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        Bt.entries, Bt.rowIndices, Bt.colIndices, Bt.shape
                )
        );
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method <i>may</i> be significantly
     * faster for large matrices as it will not make superfluous copies.
     *
     * @param exponent The exponent in the matrix power. If {@code exponent = 0} then the identity matrix will be
     *                 returned.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     * @throws IllegalArgumentException If {@code exponent} is negative.
     * @throws IllegalArgumentException If this sparse matrix is not square.
     */
    @Override
    public Matrix pow(int exponent) {
        ParameterChecks.assertSquareMatrix(shape);
        ParameterChecks.assertGreaterEq(0, exponent);

        Matrix power;

        if(exponent==0) {
            power = Matrix.I(numRows);
        } else if(exponent==1) {
            power = this.toDense();
        } else {
            // Compute the first sparse-sparse matrix multiplication.
            double[] destEntries = RealSparseMatrixMultiplication.standard(
                    entries, rowIndices, colIndices, shape,
                    entries, rowIndices, colIndices, shape
            );

            // Compute the remaining dense-sparse matrix multiplications.
            for(int i=2; i<exponent; i++) {
                destEntries = RealDenseSparseMatrixMultiplication.standard(
                        destEntries, shape,
                        entries, rowIndices, colIndices, shape
                );
            }

            power = new Matrix(shape, destEntries);
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
    public CooMatrix elemMult(Matrix B) {
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
    public CooCMatrix elemMult(CMatrix B) {
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
    public CooCMatrix elemMult(CooCMatrix B) {
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
    public CooCMatrix elemDiv(CMatrix B) {
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
        return this.T().mult(B).tr();
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vectors of length m.
     */
    @Override
    public Vector sumCols() {
        Vector sum = new Vector(this.numRows);

        int nnz = entries.length;
        for(int i=0; i<nnz; i++) {
            sum.entries[rowIndices[i]] += entries[i];
        }

        return sum;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vector of length n.
     */
    @Override
    public Vector sumRows() {
        Vector sum = new Vector(this.numCols);

        int nnz = entries.length;
        for(int i=0; i<nnz; i++) {
            sum.entries[colIndices[i]] += entries[i];
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
    public Matrix addToEachCol(CooVector b) {
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
    public CMatrix addToEachCol(CooCVector b) {
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
    public Matrix addToEachRow(CooVector b) {
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
    public CMatrix addToEachRow(CooCVector b) {
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

        Shape destShape = new Shape(numRows+B.numRows, numCols);
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
    public CooMatrix stack(CooMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numRows+B.numRows, numCols);
        double[] destEntries = new double[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        int[] shiftedRowIndices = ArrayUtils.shift(numRows, B.rowIndices.clone());
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, colIndices.length, B.colIndices.length);

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
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

        Shape destShape = new Shape(numRows+B.numRows, numCols);
        CNumber[] destEntries = new CNumber[destShape.totalEntries().intValueExact()];

        // Copy values from B
        System.arraycopy(B.entries, 0, destEntries, shape.totalEntries().intValueExact(), B.entries.length);

        // Copy non-zero values from this matrix (and set zero values to zero.).
        Arrays.fill(destEntries, 0, shape.totalEntries().intValueExact(), CNumber.ZERO);
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
    public CooCMatrix stack(CooCMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numRows+B.numRows, numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        int[] shiftedRowIndices = ArrayUtils.shift(numRows, B.rowIndices.clone());
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(B.colIndices, 0, destColIndices, colIndices.length, B.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
            destEntries[rowIndices[i]*destShape.get(1) + colIndices[i]] = entries[i];
        }

        // Copy dense values by row.
        for(int i=0; i<numRows; i++) {
            int startIdx = i*destShape.get(1) + numCols;
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
    public CooMatrix augment(CooMatrix B) {
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

        // Copy column indices (with shifts if appropriate).
        int[] shifted = B.colIndices.clone();
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(ArrayUtils.shift(numCols, shifted), 0,
                destColIndices, colIndices.length, B.colIndices.length);
        
        CooMatrix dest = new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure indices are sorted properly.

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
        Arrays.fill(destEntries, CNumber.ZERO);

        // Copy sparse values.
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*destShape.get(1) + colIndices[i]] = new CNumber(entries[i]);
        }

        // Copy dense values by row.
        for(int i=0; i<numRows; i++) {
            int startIdx = i*destShape.get(1) + numCols;
            System.arraycopy(B.entries, i*B.numCols, destEntries, startIdx, B.numCols);
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
    public CooCMatrix augment(CooCMatrix B) {
        ParameterChecks.assertEquals(numRows, B.numRows);

        Shape destShape = new Shape(numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(B.rowIndices, 0, destRowIndices, rowIndices.length, B.rowIndices.length);

        // Copy column indices (with shifts if appropriate).
        int[] shifted = B.colIndices.clone();
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(ArrayUtils.shift(numCols, shifted), 0,
                destColIndices, colIndices.length, B.colIndices.length);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure indices are sorted properly.

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
    public CooMatrix stack(Vector b) {
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
        System.arraycopy(ArrayUtils.intRange(0, numCols), 0, destColIndices, entries.length, numCols);

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooMatrix stack(CooVector b) {
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        double[] destEntries = new double[entries.length + b.entries.length];
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

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooCMatrix stack(CVector b) {
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
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.size);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(ArrayUtils.intRange(0, numCols), 0, destColIndices, entries.length, numCols);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooCMatrix stack(CooCVector b) {
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        CNumber[] destEntries = new CNumber[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy values and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(b.indices, 0, destColIndices, entries.length, b.entries.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooMatrix augment(Vector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        double[] destEntries = new double[nnz + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(ArrayUtils.intRange(0, numRows), 0, destRowIndices, entries.length, numRows);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        CooMatrix dest = new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix.<br>
     * Also see {@link #stack(CooVector)} and {@link #stack(CooVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CooMatrix augment(CooVector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        double[] destEntries = new double[nnz + b.entries.length];
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

        CooMatrix dest = new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

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
    public CooCMatrix augment(CVector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        CNumber[] destEntries = new CNumber[nnz + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(ArrayUtils.intRange(0, numRows), 0, destRowIndices, entries.length, numRows);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix.<br>
     * Also see {@link #stack(CooCVector)} and {@link #stack(CooCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CooCMatrix augment(CooCVector b) {
        ParameterChecks.assertEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        CNumber[] destEntries = new CNumber[nnz + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(b.indices, 0, destRowIndices, entries.length, b.entries.length);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);

        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    @Override
    public CooVector getRow(int i) {
        return RealSparseMatrixGetSet.getRow(this, i);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public CooVector getCol(int j) {
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
    public CooVector getCol(int colIdx, int rowStart, int rowEnd) {
        return RealSparseMatrixGetSet.getCol(this, colIdx, rowStart, rowEnd);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooVector toVector() {
        int[] destIndices = new int[indices.length];

        for(int i=0; i<entries.length; i++) {
            destIndices[i] = rowIndices[i]*colIndices[i];
        }

        return new CooVector(numRows*numCols, entries.clone(), destIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public CooTensor toTensor() {
        int[][] destIndices = RealDenseTranspose.standardIntMatrix(indices);
        return new CooTensor(this.shape, this.entries.clone(), destIndices);
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
    public CooMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
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
    public CooVector getColBelow(int rowStart, int j) {
        return RealSparseMatrixGetSet.getCol(this, j, rowStart, numRows);
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
    public CooVector getRowAfter(int colStart, int i) {
        return RealSparseMatrixGetSet.getRow(this, i, colStart, numCols);
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
    public CooMatrix setCol(CooVector values, int j) {
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
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public CooVector getDiag() {
        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                destEntries.add(entries[i]);
                destIndices.add(rowIndices[i]);
            }
        }

        return new CooVector(
                Math.min(numRows, numCols),
                ArrayUtils.fromDoubleList(destEntries),
                ArrayUtils.fromIntegerList(destIndices)
        );
    }


    /**
     * Compute the hermitian transpose of this matrix. That is, the complex conjugate transpose of this matrix.
     * For real matrices, this is equivalent to the standard transpose.
     *
     * @return The complex conjugate transpose of this matrix.
     */
    @Override
    public CooMatrix H() {
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
    public Matrix mult(CooMatrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new Matrix(numRows, B.numCols,
                RealSparseMatrixMultiplication.standard(
                    entries, rowIndices, colIndices, shape,
                    B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(CsrMatrix B) {
        return mult(B.toCoo());
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
                RealComplexDenseSparseMatrixMultiplication.standard(
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
    public CMatrix mult(CooCMatrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new CMatrix(numRows, B.numCols,
                RealComplexSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape
                )
        );
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CsrCMatrix B) {
        return mult(B.toCoo());
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
                RealDenseSparseMatrixMultiplication.standardVector(
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
        return shape.get(0)==shape.get(1);
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
    public CooCMatrix sqrtComplex() {
        return new CooCMatrix(shape, ComplexOperations.sqrt(entries), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CooCMatrix toComplex() {
        CNumber[] dest = new CNumber[entries.length];
        ArrayUtils.copy2CNumber(entries, dest);
        return new CooCMatrix(shape, dest, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CooMatrix getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooMatrix tensor, double relTol, double absTol) {
        return RealSparseEquals.allCloseMatrix(this, tensor, relTol, absTol);
    }


    /**
     * Sets an index of this tensor to a specified value. Note: Unlike with dense matrices, this will
     * return a new copy of the sparse matrix.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A copy of this matrix with the specified value set.
     * @throws IllegalArgumentException If there are not exactly two {@code indices} provided.
     */
    @Override
    public CooMatrix set(double value, int... indices) {
        ParameterChecks.assertEquals(2, indices.length);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Copies and reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     * in the reshaped matrix.
     *
     * @param newShape Shape of the new matrix.
     *
     * @return A matrix which is equivalent to this matrix but with the specified shape.
     *
     * @throws IllegalArgumentException If this matrix cannot be reshaped to the specified dimensions.
     */
    @Override
    public CooMatrix reshape(Shape newShape) {
        ParameterChecks.assertBroadcastable(shape, newShape);

        int oldColCount = shape.get(1);
        int newColCount = newShape.get(1);

        // Initialize new COO structures with the same size as the original
        int[] newRowIndices = new int[rowIndices.length];
        int[] newColIndices = new int[colIndices.length];
        double[] newEntries = new double[entries.length];

        for (int i = 0; i < rowIndices.length; i++) {
            int flatIndex = rowIndices[i]*oldColCount + colIndices[i];

            newRowIndices[i] = flatIndex / newColCount;
            newColIndices[i] = flatIndex % newColCount;
        }

        return new CooMatrix(newShape, entries.clone(), newRowIndices, newColIndices);
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public CooMatrix flatten() {
        int[] destIndices = new int[entries.length];

        for(int i = 0; i < entries.length; i++) {
            destIndices[i] = shape.entriesIndex(rowIndices[i], colIndices[i]);
        }

        return new CooMatrix(shape, entries.clone(), new int[entries.length], destIndices);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public CooMatrix flatten(int axis) {
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

        return new CooMatrix(new Shape(dims), entries.clone(), rowIndices, colIndices);
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nnz;
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




