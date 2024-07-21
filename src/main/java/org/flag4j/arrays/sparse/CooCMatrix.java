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
import org.flag4j.core.ComplexMatrixMixin;
import org.flag4j.core.MatrixMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.sparse_base.ComplexSparseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.dense.real.RealDenseTranspose;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseMatrixOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixOperations;
import org.flag4j.operations.sparse.coo.SparseDataWrapper;
import org.flag4j.operations.sparse.coo.complex.*;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexSparseMatrixMultiplication;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexSparseMatrixOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A Complex sparse matrix. Stored in coordinate list (COO) format.
 */
public class CooCMatrix
        extends ComplexSparseTensorBase<CooCMatrix, CMatrix, CooMatrix>
        implements MatrixMixin<CooCMatrix, CMatrix, CooCMatrix, CooCMatrix, CNumber, CooCVector, CVector>,
        ComplexMatrixMixin<CooCMatrix>
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
     * Creates a square sparse matrix filled with zeros.
     * @param size size of the square matrix.
     */
    public CooCMatrix(int size) {
        super(new Shape(size, size), 0, new CNumber[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix of specified size filled with zeros.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     */
    public CooCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), 0, new CNumber[0], new int[0][0]);
        rowIndices = new int[0];
        colIndices = new int[0];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix of specified shape filled with zeros.
     * @param shape Shape of this sparse matrix.
     */
    public CooCMatrix(Shape shape) {
        super(shape, 0, new CNumber[0], new int[0], new int[0]);
        rowIndices = indices[0];
        colIndices = indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a  square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public CooCMatrix(int size, CNumber[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                nonZeroEntries,
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        this.numRows = size;
        this.numCols = size;
    }


    /**
     * Creates a sparse matrix with specified size, non-zero entries, row indices, and column indices.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public CooCMatrix(int rows, int cols, CNumber[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                nonZeroEntries,
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        this.numRows = rows;
        this.numCols = cols;
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
    public CooCMatrix(Shape shape, CNumber[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                nonZeroEntries,
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a  square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public CooCMatrix(int size, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        ArrayUtils.copy2CNumber(nonZeroEntries, entries);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix with specified size, non-zero entries, row indices, and column indices.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public CooCMatrix(int rows, int cols, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        ArrayUtils.copy2CNumber(nonZeroEntries, entries);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
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
    public CooCMatrix(Shape shape, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        ArrayUtils.copy2CNumber(nonZeroEntries, entries);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Creates a  square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public CooCMatrix(int size, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        ArrayUtils.copy2CNumber(nonZeroEntries, entries);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a sparse matrix with specified size, non-zero entries, row indices, and column indices.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public CooCMatrix(int rows, int cols, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        ArrayUtils.copy2CNumber(nonZeroEntries, entries);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
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
    public CooCMatrix(Shape shape, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape,
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                rowIndices, colIndices
        );
        ParameterChecks.assertEquals(nonZeroEntries.length, rowIndices.length, colIndices.length);
        ArrayUtils.copy2CNumber(nonZeroEntries, entries);
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Creates a complex sparse matrix with specified shape, non-zero entries, and indices.
     * @param shape Shape of the sparse matrix.
     * @param entries Non-zero entries of the sparse matrix.
     * @param rowIndices Non-zero row indices of the sparse matrix.
     * @param colIndices Non-zero column indices of the sparse matrix.
     */
    public CooCMatrix(Shape shape, List<CNumber> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(
                shape,
                entries.size(),
                entries.toArray(CNumber[]::new),
                ArrayUtils.fromIntegerList(rowIndices),
                ArrayUtils.fromIntegerList(colIndices)
        );
        ParameterChecks.assertEquals(entries.size(), rowIndices.size(), colIndices.size());
        this.rowIndices = indices[0];
        this.colIndices = indices[1];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Constructs a sparse complex matrix whose non-zero entries, indices, and shape are specified by another
     * complex sparse matrix.
     * @param A Complex sparse matrix to copy.
     */
    public CooCMatrix(CooCMatrix A) {
        super(A.shape.copy(),
                A.nonZeroEntries(),
                ArrayUtils.copyOf(A.entries),
                A.rowIndices.clone(),
                A.colIndices.clone()
        );
        rowIndices = indices[0];
        colIndices = indices[1];
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Checks if an object is equal to this sparse COO matrix.
     * @param object Object to compare this sparse COO matrix to.
     * @return True if the object is a {@link CooCMatrix}, has the same shape as this matrix, and is element-wise equal to this
     * matrix.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(!(object instanceof CooCMatrix)) return false;

        CooCMatrix src2 = (CooCMatrix) object;
        return ComplexSparseEquals.matrixEquals(this, src2);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CooCMatrix getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooCMatrix tensor, double relTol, double absTol) {
        return ComplexSparseEquals.allCloseMatrix(this, tensor, relTol, absTol);
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
    public CooCMatrix add(CooCMatrix B) {
        return ComplexSparseMatrixOperations.add(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(double a) {
        return RealComplexSparseMatrixOperations.add(this, a);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        return ComplexSparseMatrixOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooCMatrix sub(CooCMatrix B) {
        return ComplexSparseMatrixOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix sub(double a) {
        return RealComplexSparseMatrixOperations.sub(this, a);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        return ComplexSparseMatrixOperations.sub(this, a);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooCMatrix transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooCMatrix T() {
        CooCMatrix transpose = new CooCMatrix(
                shape.copy().swapAxes(0, 1),
                ArrayUtils.copyOf(entries),
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
    public CNumber get(int... indices) {
        ParameterChecks.assertEquals(indices.length, 2);
        ParameterChecks.assertIndexInBounds(numRows, indices[0]);
        ParameterChecks.assertIndexInBounds(numCols, indices[1]);

        return ComplexSparseMatrixGetSet.matrixGet(this, indices[0], indices[1]);
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CooCMatrix copy() {
        return new CooCMatrix(
                shape.copy(),
                ArrayUtils.copyOf(entries),
                rowIndices.clone(),
                colIndices.clone()
        );
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooCMatrix elemMult(CooCMatrix B) {
        return ComplexSparseMatrixOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooCMatrix elemDiv(CMatrix B) {
        return ComplexDenseSparseMatrixOperations.elemDiv(this, B);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CooCMatrix makeTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CooCMatrix(shape, entries, indices[0], indices[1]);
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
    protected CooMatrix makeRealTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooMatrix(shape, entries, indices[0], indices[1]);
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
    public CMatrix toDense() {
        CNumber[] entries = new CNumber[totalEntries().intValueExact()];
        ArrayUtils.fillZeros(entries);
        int row;
        int col;

        for(int i=0; i<nonZeroEntries; i++) {
            row = rowIndices[i];
            col = colIndices[i];

            entries[row*numCols + col] = this.entries[i];
        }

        return new CMatrix(shape.copy(), entries);
    }


    /**
     * Converts this COO matrix to an equivalent CSR matrix.
     * @return An equivalent matrix in {@link CsrMatrix CSR format}.
     */
    public CsrCMatrix toCsr() {
        return new CsrCMatrix(this);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return ComplexSparseMatrixProperties.isHermation(this);
    }


    /**
     * Checks if a matrix is anti-Hermitian. That is, if the matrix is equal to the negative of its conjugate transpose.
     *
     * @return True if this matrix is antisymmetric. Otherwise, returns false.
     */
    @Override
    public boolean isAntiHermitian() {
        return ComplexSparseMatrixProperties.isAntiHermation(this);
    }


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is equal to its conjugate transpose.
     *
     * @return True if this matrix it is unitary. Otherwise, returns false.
     */
    @Override
    public boolean isUnitary() {
        if(isSquare()) {
            return mult(H()).round().equals(CMatrix.I(numRows));
        } else {
            return false;
        }
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the values vector has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex} is not within the matrix.
     */
    @Override
    public CooCMatrix setCol(CVector values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(this, colIndex, values.entries);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the values vector has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex} is not within the matrix.
     */
    public CooCMatrix setCol(Vector values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(this, colIndex, values.entries);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    @Override
    public CooCMatrix setRow(CVector values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(this, rowIndex, values.entries);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    @Override
    public CooCMatrix setRow(CooCVector values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(this, rowIndex, values);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    public CooCMatrix setRow(Vector values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(this, rowIndex, values.entries);
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CooCMatrix hermTranspose() {
        return H();
    }


    /**
     * Computes the element-wise addition between two matrices.
     *
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(Matrix B) {
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
    public CooCMatrix add(CooMatrix B) {
        return RealComplexSparseMatrixOperations.add(this, B);
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
        return ComplexDenseSparseMatrixOperations.add(B, this);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(Matrix B) {
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
    public CooCMatrix sub(CooMatrix B) {
        return RealComplexSparseMatrixOperations.sub(this, B);
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
        return ComplexDenseSparseMatrixOperations.sub(this, B);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  rows in matrix {@code B}.
     */
    @Override
    public CMatrix mult(Matrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);
        CNumber[] dest = RealComplexDenseSparseMatrixMultiplication.standard(
                entries, rowIndices, colIndices, shape,
                B.entries, B.shape
        );
        return new CMatrix(new Shape(numRows, B.numCols), dest);
    }


    /**
     * Computes the matrix-vector multiplication.
     *
     * @param b Vector to multiply this matrix to.
     * @return The vector result from multiplying this matrix by the vector {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  entries {@code B}.
     */
    @Override
    public CVector mult(CooCVector b) {
        ParameterChecks.assertEquals(numCols, b.size);
        CNumber[] dest = ComplexSparseMatrixMultiplication.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices
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
    public CMatrix multTranspose(Matrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        Matrix Bt = TransposeDispatcher.dispatch(B);

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
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CooMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        CooMatrix Bt = B.T();

        return new CMatrix(
                numRows, Bt.numCols,
                RealComplexSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        Bt.entries, Bt.rowIndices, Bt.colIndices, Bt.shape
                )
        );
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
        ParameterChecks.assertEquals(numCols, B.numCols);
        CMatrix Bt = TransposeDispatcher.dispatch(B);

        return new CMatrix(
                numRows, Bt.numCols,
                ComplexDenseSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        Bt.entries, Bt.shape
                )
        );
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
    public CMatrix multTranspose(CooCMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);
        CooCMatrix Bt = B.T();

        return new CMatrix(
                numRows, Bt.numCols,
                ComplexSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        Bt.entries, Bt.rowIndices, Bt.colIndices, Bt.shape
                )
        );
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method <i>may</i> be significantly
     * faster.
     *
     * @param exponent The exponent in the matrix power.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     */
    @Override
    public CMatrix pow(int exponent) {
        ParameterChecks.assertSquareMatrix(shape);
        ParameterChecks.assertGreaterEq(0, exponent);

        CMatrix power;

        if(exponent==0) {
            power = CMatrix.I(numRows);
        } else if(exponent==1) {
            power = this.toDense();
        } else {
            // Compute the first sparse-sparse matrix multiplication.
            CNumber[] destEntries = ComplexSparseMatrixMultiplication.standard(
                    entries, rowIndices, colIndices, shape,
                    entries, rowIndices, colIndices, shape
            );

            // Compute the remaining dense-sparse matrix multiplications.
            for(int i=2; i<exponent; i++) {
                destEntries = ComplexDenseSparseMatrixMultiplication.standard(
                        destEntries, shape,
                        entries, rowIndices, colIndices, shape
                );
            }

            power = new CMatrix(shape.copy(), destEntries);
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
    public CooCMatrix elemMult(Matrix B) {
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
    public CooCMatrix elemMult(CooMatrix B) {
        return RealComplexSparseMatrixOperations.elemMult(this, B);
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
        return ComplexDenseSparseMatrixOperations.elemMult(B, this);
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
    public CooCMatrix elemDiv(Matrix B) {
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
    public CNumber det() {
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
    public CNumber fib(Matrix B) {
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
    public CNumber fib(CooMatrix B) {
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
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vectors of length m.
     */
    @Override
    public CVector sumCols() {
        CVector sum = new CVector(numRows);

        int nnz = entries.length;
        for(int i=0; i<nnz; i++) {
            sum.entries[rowIndices[i]].addEq(entries[i]);
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
    public CVector sumRows() {
        CVector sum = new CVector(numCols);

        int nnz = entries.length;
        for(int i=0; i<nnz; i++) {
            sum.entries[colIndices[i]].addEq(entries[i]);
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
    public CMatrix addToEachCol(Vector b) {
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
    public CMatrix addToEachCol(CooVector b) {
        return RealComplexSparseMatrixOperations.addToEachCol(this, b);
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
        return ComplexDenseSparseMatrixOperations.addToEachCol(this, b);
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
        return ComplexSparseMatrixOperations.addToEachCol(this, b);
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(Vector b) {
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
    public CMatrix addToEachRow(CooVector b) {
        return RealComplexSparseMatrixOperations.addToEachRow(this, b);
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
        return ComplexDenseSparseMatrixOperations.addToEachRow(this, b);
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
        return ComplexSparseMatrixOperations.addToEachRow(this, b);
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
    public CMatrix stack(Matrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numRows+B.numRows, numCols);
        CNumber[] destEntries = new CNumber[destShape.totalEntries().intValueExact()];
        ArrayUtils.fillZeros(destEntries);

        // Copy values from B
        ArrayUtils.arraycopy(B.entries, 0, destEntries, shape.totalEntries().intValueExact(), B.entries.length);

        // Copy non-zero values from this matrix.
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*numCols + colIndices[i]] = entries[i].copy();
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
    public CooCMatrix stack(CooMatrix B) {
        ParameterChecks.assertEquals(numCols, B.numCols);

        Shape destShape = new Shape(numRows+B.numRows, numCols);
        CNumber[] destEntries = new CNumber[entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

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
        ArrayUtils.arraycopy(B.entries, 0, destEntries, shape.totalEntries().intValueExact(), B.entries.length);

        // Copy non-zero values from this matrix (and set zero values to zero.).
        ArrayUtils.fillZeros(destEntries, 0, shape.totalEntries().intValueExact());
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*numCols + colIndices[i]] = entries[i].copy();
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
        ArrayUtils.arraycopy(B.entries, 0, destEntries, entries.length, B.entries.length);

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
    public CMatrix augment(Matrix B) {
        ParameterChecks.assertEquals(numRows, B.numRows);

        Shape destShape = new Shape(numRows, numCols + B.numCols);
        CNumber[] destEntries = new CNumber[destShape.totalEntries().intValueExact()];
        ArrayUtils.fillZeros(destEntries);

        // Copy sparse values.
        for(int i=0; i<entries.length; i++) {
            destEntries[rowIndices[i]*destShape.dims[1] + colIndices[i]] = entries[i].copy();
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
    public CooCMatrix augment(CooMatrix B) {
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
        ArrayUtils.fillZeros(destEntries);

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
    public CooCMatrix augment(CooCMatrix B) {
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
    public CooCMatrix stack(Vector b) {
        ParameterChecks.assertEquals(numCols, b.size);

        Shape destShape = new Shape(numRows + 1, numCols);
        CNumber[] destEntries = new CNumber[entries.length + b.size];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        ArrayUtils.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy values from vector and create indices.
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.size);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(ArrayUtils.intRange(0, numCols), 0, destColIndices, entries.length, numCols);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooCMatrix stack(CooVector b) {
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
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(b.indices, 0, destColIndices, entries.length, b.entries.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.size);
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
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(b.indices, 0, destColIndices, entries.length, b.entries.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooCMatrix augment(Vector b) {
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
        System.arraycopy(ArrayUtils.intRange(0, numRows), 0, destRowIndices, entries.length, numRows);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
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
    public CooCMatrix augment(CooVector b) {
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

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
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
    public CooCMatrix augment(CVector b) {
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
        System.arraycopy(ArrayUtils.intRange(0, numRows), 0, destRowIndices, entries.length, numRows);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
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
    public CooCMatrix augment(CooCVector b) {
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
    public CooCVector getRow(int i) {
        return ComplexSparseMatrixGetSet.getRow(this, i);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public CooCVector getCol(int j) {
        return ComplexSparseMatrixGetSet.getCol(this, j);
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
    public CooCVector getCol(int colIdx, int rowStart, int rowEnd) {
        return ComplexSparseMatrixGetSet.getCol(this, colIdx, rowStart, rowEnd);
    }


    /**
     * Converts this matrix to an equivalent complex tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public CooCTensor toTensor() {
        int[][] destIndices = RealDenseTranspose.standardIntMatrix(indices);
        return new CooCTensor(this.shape.copy(), ArrayUtils.copyOf(entries), destIndices);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooCVector toVector() {
        int[] destIndices = new int[indices.length];

        for(int i=0; i<entries.length; i++) {
            destIndices[i] = rowIndices[i]*colIndices[i];
        }

        return new CooCVector(numRows*numCols, ArrayUtils.copyOf(entries), destIndices);
    }


    @Override
    public CooMatrix toReal() {
        return new CooMatrix(shape.copy(), ComplexOperations.toReal(entries), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Constructs a sparse COO matrix from a dense matrix. Any value that is not exactly zero will be considered a non-zero value.
     * @param src Dense matrix to convert to sparse COO matrix.
     * @return An sparse COO matrix equivalent to the dense {@code src} matrix.
     */
    public static CooCMatrix fromDense(CMatrix src) {
        int rows = src.numRows;
        int cols = src.numCols;
        List<CNumber> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                CNumber val = src.entries[rowOffset + j];

                if(!val.equals(0)) {
                    entries.add(val.copy());
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooCMatrix(src.shape.copy(), entries, rowIndices, colIndices);
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
    public CooCMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        return ComplexSparseMatrixGetSet.getSlice(this, rowStart, rowEnd, colStart, colEnd);
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
    public CooCVector getColBelow(int rowStart, int j) {
        return ComplexSparseMatrixGetSet.getCol(this, j, rowStart, numRows);
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
    public CooCVector getRowAfter(int colStart, int i) {
        return ComplexSparseMatrixGetSet.getRow(this, i, colStart, numCols);
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
    public CooCMatrix setCol(CooCVector values, int j) {
        return ComplexSparseMatrixGetSet.setCol(this, j, values);
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #tr()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public CNumber trace() {
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
    public CNumber tr() {
        CNumber trace = new CNumber();

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                trace.addEq(entries[i]);
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
    public CooCVector getDiag() {
        List<CNumber> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                destEntries.add(entries[i]);
                destIndices.add(rowIndices[i]);
            }
        }

        CNumber[] destArr = new CNumber[destEntries.size()];

        return new CooCVector(
                numRows,
                ArrayUtils.fromList(destEntries, destArr),
                ArrayUtils.fromIntegerList(destIndices)
        );
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  rows in matrix {@code B}.
     */
    @Override
    public CMatrix mult(CooCMatrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new CMatrix(numRows, B.numCols,
                ComplexSparseMatrixMultiplication.standard(
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
    public CVector mult(Vector b) {
        ParameterChecks.assertMatMultShapes(shape, new Shape(b.size, 1));

        return new CVector(
                RealComplexDenseSparseMatrixMultiplication.standardVector(
                        this.entries, this.rowIndices, this.colIndices, this.shape,
                        b.entries, b.shape
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
    public CVector mult(CooVector b) {
        CNumber[] product = RealComplexSparseMatrixMultiplication.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices, b.shape
        );

        return new CVector(product);
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
        ParameterChecks.assertEquals(numCols, b.size);

        CNumber[] product = ComplexDenseSparseMatrixMultiplication.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.shape
        );

        return new CVector(product);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CooMatrix B) {
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
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CMatrix B) {
        ParameterChecks.assertMatMultShapes(shape, B.shape);

        return new CMatrix(numRows, B.numCols,
                ComplexDenseSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        B.entries, B.shape
                )
        );
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()} and
     * {@link #conjT()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CooCMatrix H() {
        CooCMatrix hTranspose = new CooCMatrix(
                shape.copy().swapAxes(0, 1),
                ComplexOperations.conj(entries),
                colIndices.clone(),
                rowIndices.clone()
        );

        hTranspose.sortIndices(); // Ensure the indices are sorted correctly.

        return hTranspose;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException  If the number of indices is not equal to the rank of this tensor.
     * @throws IndexOutOfBoundsException If any of the indices are not within this tensor.
     */
    @Override
    public CooCMatrix set(CNumber value, int... indices) {
        ParameterChecks.assertEquals(indices.length, 2);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public CooCMatrix flatten(int axis) {
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

        return new CooCMatrix(new Shape(dims), ArrayUtils.copyOf(entries), rowIndices, colIndices);
    }



    /**
     * Checks if this matrix is the identity matrix.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        return ComplexSparseMatrixProperties.isIdentity(this);
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
    public CooCMatrix set(double value, int row, int col) {
        return set(new CNumber(value), row, col);
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
    public CooCMatrix set(CNumber value, int row, int col) {
        ParameterChecks.assertIndexInBounds(numRows, row);
        ParameterChecks.assertIndexInBounds(numCols, col);

        return ComplexSparseMatrixGetSet.matrixSet(this, row, col, value);
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
    public CooCMatrix setCol(CNumber[] values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(this, colIndex, values);
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
    public CooCMatrix setCol(Double[] values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(this, colIndex, ArrayUtils.unbox(values));
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
    public CooCMatrix setCol(Integer[] values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(
                this,
                colIndex,
                ArrayUtils.copy2CNumber(values, null)
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
    public CooCMatrix setCol(double[] values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(
                this, colIndex, values
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
    public CooCMatrix setCol(int[] values, int colIndex) {
        return ComplexSparseMatrixGetSet.setCol(
                this,
                colIndex,
                ArrayUtils.copy2CNumber(values, null)
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
    public CooCMatrix setRow(CNumber[] values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(this, rowIndex, values);
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
    public CooCMatrix setRow(Double[] values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(this, rowIndex, ArrayUtils.unbox(values));
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
    public CooCMatrix setRow(Integer[] values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(
                this, rowIndex, ArrayUtils.copy2CNumber(values, null)
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
    public CooCMatrix setRow(double[] values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(this, rowIndex, values);
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
    public CooCMatrix setRow(int[] values, int rowIndex) {
        return ComplexSparseMatrixGetSet.setRow(
                this,
                rowIndex,
                ArrayUtils.copy2CNumber(values, null)
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
    public CooCMatrix setSlice(CooCMatrix values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(Matrix values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(CooMatrix values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(CNumber[][] values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(Double[][] values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(Integer[][] values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(double[][] values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooCMatrix setSlice(int[][] values, int rowStart, int colStart) {
        return ComplexSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     * @return a copy of this matrix with the specified row removed.
     */
    @Override
    public CooCMatrix removeRow(int rowIndex) {
        return ComplexSparseMatrixManipulations.removeRow(this, rowIndex);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix.
     * @return a copy of this matrix with the specified rows removed.
     */
    @Override
    public CooCMatrix removeRows(int... rowIndices) {
        return ComplexSparseMatrixManipulations.removeRows(this, rowIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooCMatrix removeCol(int colIndex) {
        return ComplexSparseMatrixManipulations.removeCol(this, colIndex);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return a copy of this matrix with the specified columns removed.
     */
    @Override
    public CooCMatrix removeCols(int... colIndices) {
        return ComplexSparseMatrixManipulations.removeCols(this, colIndices);
    }


    /**
     * Swaps rows in the matrix.
     *
     * @param rowIndex1 Index of first row to swap.
     * @param rowIndex2 index of second row to swap.
     * @return A reference to this matrix.
     */
    @Override
    public CooCMatrix swapRows(int rowIndex1, int rowIndex2) {
        return ComplexSparseMatrixManipulations.swapRows(this, rowIndex1, rowIndex2);
    }


    /**
     * Swaps columns in the matrix.
     *
     * @param colIndex1 Index of first column to swap.
     * @param colIndex2 index of second column to swap.
     * @return A reference to this matrix.
     */
    @Override
    public CooCMatrix swapCols(int colIndex1, int colIndex2) {
        return ComplexSparseMatrixManipulations.swapCols(this, colIndex1, colIndex2);
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
        return numCols==1 || numRows==1;
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
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return toDense().isFullRank();
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     *
     * @return True if this matrix is singular. Otherwise, returns false.
     */
    @Override
    public boolean isSingular() {
        return toDense().isSingular();
    }


    /**
     * Checks if a matrix is invertible.<br>
     * Also see {@link #isSingular()}.
     *
     * @return True if this matrix is invertible.
     */
    @Override
    public boolean isInvertible() {
        return !isSingular();
    }


    /**
     * Computes the rank of this matrix (i.e. the dimension of the column space of this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank.
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        return toDense().matrixRank();
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public CooCMatrix set(double value, int... indices) {
        ParameterChecks.assertEquals(2, indices.length);
        return set(value, indices[0], indices[1]);
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
