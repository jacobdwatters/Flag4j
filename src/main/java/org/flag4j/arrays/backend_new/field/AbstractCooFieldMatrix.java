/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.arrays.backend_new.field;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.AbstractTensor;
import org.flag4j.arrays.backend_new.MatrixMixin;
import org.flag4j.arrays.backend_new.SparseMatrixData;
import org.flag4j.arrays.backend_new.SparseVectorData;
import org.flag4j.linalg.operations.common.field_ops.FieldOps;
import org.flag4j.linalg.operations.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.operations.sparse.SparseElementSearch;
import org.flag4j.linalg.operations.sparse.coo.*;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldMatrixProperties;
import org.flag4j.linalg.operations.sparse.coo.ring_ops.CooRingMatrixOps;
import org.flag4j.linalg.operations.sparse.coo.semiring_ops.CooSemiringMatMult;
import org.flag4j.linalg.operations.sparse.coo.semiring_ops.CooSemiringMatrixOps;
import org.flag4j.linalg.operations.sparse.coo.semiring_ops.CooSemiringMatrixProperties;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.flag4j.linalg.operations.sparse.SparseUtils.copyRanges;


/**
 * <p>A sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
 * elements of a {@link Field}.</p>
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero values.</p>
 *
 * <p>COO matrices are optimized for hyper-sparse matrices (i.e. matrices which contain almost all zeros relative to the size of the
 * matrix).</p>
 *
 * <p>A sparse COO matrix is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified but any operations implemented in this
 * class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.</p>
 *
 * @param <T> Type of this sparse COO matrix.
 * @param <U> Type of dense matrix which is similar to {@code T}.
 * @param <V> Type of sparse COO vector which is similar to {@code T}.
 * @param <W> Type of the field element in this matrix.
 */
public abstract class AbstractCooFieldMatrix<T extends AbstractCooFieldMatrix<T, U, V, W>,
        U extends AbstractDenseFieldMatrix<U, ?, W>,
        V extends AbstractCooFieldVector<V, ?, T, U, W>,
        W extends Field<W>>
        extends AbstractTensor<T, Field<W>[], W>
        implements FieldTensorMixin<T, U, W>, MatrixMixin<T, U, V, W> {

    /**
     * The zero element for the field that this tensor's elements belong to.
     */
    private Field<W> zeroElement;
    /**
     * Row indices for non-zero value of this sparse COO matrix.
     */
    public final int[] rowIndices;
    /**
     * column indices for non-zero value of this sparse COO matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero entries in this COO matrix.
     */
    public final int nnz;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;
    /**
     * The sparsity of this matrix.
     */
    public final double sparsity;


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    protected AbstractCooFieldMatrix(Shape shape, Field<W>[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        ValidateParameters.ensureIndexInBounds(shape.get(0), rowIndices);
        ValidateParameters.ensureIndexInBounds(shape.get(1), colIndices);
        ValidateParameters.ensureArrayLengthsEq(entries.length, rowIndices.length, colIndices.length);

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
        sparsity = BigDecimal.valueOf(nnz).divide(new BigDecimal(shape.totalEntries())).doubleValue();

        // Attempt to set the zero element for the field.
        this.zeroElement = (entries.length > 0) ? entries[0].getZero() : null;
    }


    /**
     * Constructs a sparse COO tensor of the same type as this tensor with the specified non-zero entries and indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return A sparse COO tensor of the same type as this tensor with the specified non-zero entries and indices.
     */
    public abstract T makeLikeTensor(Shape shape, W[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Constructs a COO matrix with the specified shape, non-zero entries, and non-zero indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return A COO matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    public abstract T makeLikeTensor(Shape shape, List<Field<W>> entries, List<Integer> rowIndices, List<Integer> colIndices);


    /**
     * Constructs a sparse COO vector of a similar type to this COO matrix.
     * @param shape Shape of the vector. Must be rank 1.
     * @param entries Non-zero entries of the COO vector.
     * @param indices Non-zero indices of the COO vector.
     * @return A sparse COO vector of a similar type to this COO matrix.
     */
    public abstract V makeLikeVector(Shape shape, Field<W>[] entries, int[] indices);


    /**
     * Constructs a dense tensor with the specified {@code shape} and {@code entries} which is a similar type to this sparse tensor.
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     * @return A dense tensor with the specified {@code shape} and {@code entries} which is a similar type to this sparse tensor.
     */
    public abstract U makeLikeDenseTensor(Shape shape, Field<W>[] entries);


    /**
     * Constructs a sparse CSR matrix of a similar type to this sparse COO matrix.
     * @param shape Shape of the CSR matrix to construct.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A CSR matrix of a similar type to this sparse COO matrix.
     */
    public abstract AbstractCsrFieldMatrix<?, U, V, W> makeLikeCsrMatrix(
            Shape shape, Field<W>[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Gets the zero element for the field of this tensor.
     * @return The zero element for the field of this tensor. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Field)} then {@code null} will be returned.
     *
     * @see #setZeroElement(Field)
     */
    public W getZeroElement() {
        return (W) zeroElement;
    }


    /**
     * Gets the sparsity of this matrix as a decimal percentage.
     * That is, the percentage of entries in this matrix that are zero.
     * @return The sparsity of this matrix as a decimal percentage.
     * @see #density()
     */
    public double sparsity() {
        return sparsity;
    }


    /**
     * Gets the density of this matrix as a decimal percentage.
     * That is, the percentage of entries in this matrix that are non-zero.
     * @return The density of this matrix as a decimal percentage.
     * @see #sparsity
     */
    public double density() {
        return 1.0 - sparsity;
    }


    /**
     * Sets the zero element for the field of this tensor.
     * @param zeroElement The zero element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the field.
     *
     * @see #getZeroElement()
     */
    public void setZeroElement(Field<W> zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }


    /**
     * Gets the element of this tensor at the specified index.
     *
     * @param index Indices of the element to get.
     *
     * @return The element of this tensor at the specified index. If there is a non-zero value with the specified index, that value
     * will be returned. If there is no non-zero value at the specified index than the zero element will attempt to be
     * returned (i.e. the additive identity of the field). However, if the zero element could not be determined during
     * construction or if it was not set with {@link #setZeroElement(Field)} then
     * {@code null} will be returned.
     *
     * @throws ArrayIndexOutOfBoundsException If any index are not within this tensor.
     */
    @Override
    public W get(int... index) {
        ValidateParameters.validateTensorIndex(shape, index);
        W value = (W) CooGetSet.getCoo(entries, rowIndices, colIndices, index[0], index[1]);
        return (value == null) ? getZeroElement() : value;
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(W value, int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     *
     * @return A reference to this matrix.
     */
    @Override
    public T set(W value, int row, int col) {
        // Find position of row index within the row indices if it exits.
        int idx = SparseElementSearch.matrixBinarySearch(rowIndices, colIndices, row, col);
        Field<W>[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
            idx = -idx - 1;

            // No non-zero element with these indices exists. Insert new value.
            destEntries = new Field[entries.length + 1];
            destRowIndices = new int[entries.length + 1];
            destColIndices = new int[entries.length + 1];

            CooGetSet.cooInsertNewValue(
                    value, row, col,
                    entries, rowIndices, colIndices, idx,
                    destEntries, destRowIndices, destColIndices);
        } else {
            // Value with these indices exists. Simply update value.
            destEntries = Arrays.copyOf(entries, entries.length);
            destEntries[idx] = value;
            destRowIndices = rowIndices.clone();
            destColIndices = colIndices.clone();
        }

        return makeLikeTensor(shape, (W[]) destEntries, destRowIndices, destColIndices);
    }


    /**
     * Flattens this matrix to a single row.
     *
     * @return The flattened matrix.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return flatten(0);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = {1, 1};
        dims[1-axis] = shape.totalEntriesIntValueExact();
        Shape flatShape = new Shape(dims);

        int[] destIndices = new int[entries.length];

        for(int i = 0; i < entries.length; i++)
            destIndices[i] = shape.getFlatIndex(rowIndices[i], colIndices[i]);

        return (axis == 0)
                ? makeLikeTensor(flatShape, (W[]) entries.clone(), new int[entries.length], destIndices)
                : makeLikeTensor(flatShape, (W[]) entries.clone(), destIndices, new int[entries.length]);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        ValidateParameters.ensureBroadcastable(shape, newShape);
        int oldColCount = shape.get(1);
        int newColCount = newShape.get(1);

        // Initialize new COO structures with the same size as the original.
        int[] newRowIndices = new int[rowIndices.length];
        int[] newColIndices = new int[colIndices.length];

        for (int i = 0; i < rowIndices.length; i++) {
            int flatIndex = rowIndices[i]*oldColCount + colIndices[i];
            newRowIndices[i] = flatIndex / newColCount;
            newColIndices[i] = flatIndex % newColCount;
        }

        return makeLikeTensor(newShape, (W[]) entries.clone(), newRowIndices, newColIndices);
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public T T() {
        T transpose = makeLikeTensor(shape.swapAxes(0, 1), (W[]) entries.clone(), colIndices.clone(), rowIndices.clone());
        transpose.sortIndices(); // Ensure the indices are sorted correctly.
        return transpose;
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public T T(int axis1, int axis2) {
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return copy();
        else return T();
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public T T(int... axes) {
        if(axes.length != 2)
            throw new IllegalArgumentException("Expecting two axes in transpose but got " + axes.length + ".");
        return T(axes[0], axes[1]);
    }


    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    @Override
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     *
     * @return The number of columns in this matrix.
     */
    @Override
    public int numCols() {
        return numCols;
    }


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.</p>
     *
     * <p>Same as {@link #trace()}.</p>
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public W tr() {
        W trace = getZeroElement();

        for(int i=0; i<entries.length; i++)
            if(rowIndices[i]==colIndices[i]) trace = trace.add((W) entries[i]); // Then entry is on the diagonal.

        return trace;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     *
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        for(int i=0; i<entries.length; i++)
            if(rowIndices[i] > colIndices[i] && !entries[i].isZero()) return false; // Then non-zero entry is not in upper triangle.

        return true;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     *
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        for(int i=0; i<entries.length; i++)
            if(rowIndices[i] < colIndices[i] && !entries[i].isZero()) return false; // Then non-zero entry is not in lower triangle.

        return true;
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        return CooSemiringMatrixProperties.isIdentity(shape, entries, rowIndices, colIndices);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of rows in matrix {@code b}.
     */
    @Override
    public U mult(T b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Field<W>[] dest = new Field[numRows*b.numCols];
        CooSemiringMatMult.standard(
                entries, rowIndices, colIndices, shape,
                b.entries, b.rowIndices, b.colIndices, b.shape, dest);

        return makeLikeDenseTensor(new Shape(numRows, b.numCols), dest);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public U multTranspose(T b) {
        ValidateParameters.ensureEquals(numCols, b.numCols);
        return mult(b.T());
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(MatrixMixin, int)
     * @see #augment(T)
     */
    @Override
    public T stack(T b) {
        ValidateParameters.ensureEquals(numCols, b.numCols);

        Shape destShape = new Shape(numRows+b.numRows, numCols);
        Field<W>[] destEntries = new Field[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];
        CooConcat.stack(entries, rowIndices, colIndices, numRows,
                b.entries, b.rowIndices, b.colIndices,
                destEntries, destRowIndices, destColIndices);

        return makeLikeTensor(destShape, (W[]) destEntries, destRowIndices, destColIndices);
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(T)
     * @see #stack(MatrixMixin, int)
     */
    @Override
    public T augment(T b) {
        ValidateParameters.ensureEquals(numRows, b.numRows);

        Shape destShape = new Shape(numRows, numCols + b.numCols);
        Field<W>[] destEntries = new Field[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];
        CooConcat.augment(entries, rowIndices, colIndices, numCols,
                b.entries, b.rowIndices, b.colIndices,
                destEntries, destRowIndices, destColIndices);

        return makeLikeTensor(destShape, (W[]) destEntries, destRowIndices, destColIndices);
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public T augment(V b) {
        ValidateParameters.ensureEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        Field<W>[] destEntries = new Field[nnz + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];
        CooConcat.augmentVector(
                entries, rowIndices, colIndices, numCols,
                b.entries, b.indices,
                destEntries, destRowIndices, destColIndices);

        return makeLikeTensor(destShape, (W[]) destEntries, destRowIndices, destColIndices);
    }


    /**
     * Swaps specified rows in the matrix. This is done in place.
     *
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public T swapRows(int rowIndex1, int rowIndex2) {
        CooManipulations.swapRows(shape, entries, rowIndices, colIndices, rowIndex1, rowIndex2);
        return (T) this;
    }


    /**
     * Swaps specified columns in the matrix. This is done in place.
     *
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public T swapCols(int colIndex1, int colIndex2) {
        CooManipulations.swapCols(shape, entries, rowIndices, colIndices, colIndex1, colIndex2);
        return (T) this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        return CooSemiringMatrixProperties.isSymmetric(shape, entries, rowIndices, colIndices);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return CooFieldMatrixProperties.isHermitian(shape, entries, rowIndices, colIndices);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        if(isSquare()) return mult(T()).isI();
        else return false;
    }


    /**
     * Gets a range of a row of this matrix.
     *
     * @param rowIdx The index of the row to get.
     * @param start The staring column of the row range to get (inclusive).
     * @param stop The ending column of the row range to get (exclusive).
     *
     * @return A vector containing the elements of the specified row over the range [start, stop).
     *
     * @throws IllegalArgumentException If {@code rowIdx < 0 || rowIdx >= this.numRows()} or {@code start < 0 || start >= numCols} or
     *                                  {@code stop < start || stop > numCols}.
     */
    @Override
    public V getRow(int rowIdx, int start, int stop) {
        SparseVectorData<Field<W>> data = CooGetSet.getRow(shape, entries, rowIndices, colIndices, rowIdx, start, stop);
        return (V) makeLikeVector(data.shape(),
                data.entries().toArray(new Field[data.entries().size()]),
                data.indicesToArray());
    }


    /**
     * Gets a range of a column of this matrix.
     *
     * @param colIdx The index of the column to get.
     * @param start The staring row of the column range to get (inclusive).
     * @param stop The ending row of the column range to get (exclusive).
     *
     * @return A vector containing the elements of the specified column over the range [start, stop).
     *
     * @throws IllegalArgumentException If {@code colIdx < 0 || colIdx >= this.numCols()} or {@code start < 0 || start >= numRows} or
     *                                  {@code stop < start || stop > numRows}.
     */
    @Override
    public V getCol(int colIdx, int start, int stop) {
        SparseVectorData<Field<W>> data = CooGetSet.getCol(shape, entries, rowIndices, colIndices, colIdx, start, stop);
        return (V) makeLikeVector(data.shape(),
                data.entries().toArray(new Field[data.entries().size()]),
                data.indicesToArray());
    }


    /**
     * Gets the elements of this matrix along the specified diagonal.
     *
     * @param diagOffset The diagonal to get within this matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     *
     * @return The elements of the specified diagonal as a vector.
     */
    @Override
    public V getDiag(int diagOffset) {
        SparseVectorData<Field<W>> data = CooGetSet.getDiag(shape, entries, rowIndices, colIndices, diagOffset);
        return (V) makeLikeVector(data.shape(),
                data.entries().toArray(new Field[data.entries().size()]),
                data.indicesToArray());
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public T removeRow(int rowIndex) {
        Shape shape = new Shape(numRows-1, numCols);

        // Find the start and end index within the entries array which have the given row index.
        int[] startEnd = SparseElementSearch.matrixFindRowStartEnd(rowIndices, rowIndex);
        int size = entries.length - (startEnd[1]-startEnd[0]);

        // Initialize arrays.
        Field<W>[] entries = new Field[size];
        int[] rowIndices = new int[size];
        int[] colIndices = new int[size];

        copyRanges(this.entries, this.rowIndices, this.colIndices, entries, rowIndices, colIndices, startEnd);

        return makeLikeTensor(shape, (W[]) entries, rowIndices, colIndices);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIdxs The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIdxs) {
        // TODO: This should be doable for a general COO matrix. Return SparseMatrixData object.
        Shape shape = new Shape(numRows-rowIdxs.length, numCols);
        int initSize = Math.max(0, nnz - (nnz / numRows) * rowIdxs.length); // Estimate the number of non-zeros in the result.
        List<Field<W>> entries = new ArrayList<>(initSize);
        List<Integer> rowIndices = new ArrayList<>(initSize);
        List<Integer> colIndices = new ArrayList<>(initSize);

        for(int i=0; i<nnz; i++) {
            if(!ArrayUtils.contains(rowIdxs, this.rowIndices[i])) {
                // Then copy the entry over.
                entries.add(this.entries[i]);
                rowIndices.add(this.rowIndices[i]);
                colIndices.add(this.colIndices[i]);
            }
        }

        return makeLikeTensor(shape, entries, rowIndices, colIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCol(int colIndex) {
        Shape shape = new Shape(numRows, numCols-1);
        List<Field<W>> destEntries = new ArrayList<>(entries.length);
        List<Integer> destRowIndices = new ArrayList<>(entries.length);
        List<Integer> destColIndices = new ArrayList<>(entries.length);

        for(int i=0; i<entries.length; i++) {
            if(colIndices[i] != colIndex) {
                // Then entry is not in the specified column, so remove it.
                destEntries.add(entries[i]);
                destRowIndices.add(rowIndices[i]);

                if(colIndices[i] < colIndex) destColIndices.add(colIndices[i]);
                else destColIndices.add(colIndices[i]-1);
            }
        }

        return makeLikeTensor(shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIdxs Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCols(int... colIdxs) {
        Shape shape = new Shape(numRows, numCols-1);
        List<Field<W>> destEntries = new ArrayList<>(entries.length);
        List<Integer> destRowIndices = new ArrayList<>(entries.length);
        List<Integer> destColIndices = new ArrayList<>(entries.length);

        for(int i=0; i<entries.length; i++) {
            int idx = Arrays.binarySearch(colIdxs, colIndices[i]);

            if(idx < 0) {
                // Then entry is not in the specified column, so copy it with the appropriate column index shift.
                destEntries.add(entries[i]);
                destRowIndices.add(rowIndices[i]);
                destColIndices.add(colIndices[i] + (idx+1));
            }
        }

        return makeLikeTensor(shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A copy of this matrix with the given slice set to the specified values.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public T setSliceCopy(T values, int rowStart, int colStart) {
        SparseMatrixData<Field<W>> sliceData = CooGetSet.setSlice(
                shape, entries, rowIndices, colIndices,
                values.shape, values.entries, values.rowIndices, values.colIndices,
                rowStart, colStart);
        return (T) makeLikeTensor(sliceData.shape(), sliceData.entries(), sliceData.rowIndices(), sliceData.colIndices());
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     *
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public T getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        SparseMatrixData<Field<W>> sliceData = CooGetSet.getSlice(
                shape, entries, rowIndices, colIndices,
                rowStart, rowEnd, colStart, colEnd);
        return (T) makeLikeTensor(sliceData.shape(), sliceData.entries(), sliceData.rowIndices(), sliceData.colIndices());
    }


    /**
     * Extracts the upper-triangular portion of this matrix with a specified diagonal offset. All other entries of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for upper-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all entries at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all entries at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all entries at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The upper-triangular portion of this matrix with a specified diagonal offset. All other entries of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public T getTriU(int diagOffset) {
        SparseMatrixData<Field<W>> data = CooGetSet.getTriU(diagOffset, shape, entries, rowIndices, colIndices);
        return makeLikeTensor(data.shape(),  data.entries(), data.rowIndices(), data.colIndices());
    }


    /**
     * Extracts the lower-triangular portion of this matrix with a specified diagonal offset. All other entries of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for lower-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all entries at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all entries at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all entries at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The lower-triangular portion of this matrix with a specified diagonal offset. All other entries of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public T getTriL(int diagOffset) {
        SparseMatrixData<Field<W>> data = CooGetSet.getTriL(diagOffset, shape, entries, rowIndices, colIndices);
        return makeLikeTensor(data.shape(),  data.entries(), data.rowIndices(), data.colIndices());
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, entries);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        SparseMatrixData<Ring<W>> data = CooRingMatrixOps.sub(
                shape, entries, rowIndices, colIndices,
                b.shape, b.entries, b.rowIndices, b.colIndices);

        return makeLikeTensor(data.shape(),
                (W[]) data.entries().toArray(new Field[data.entries().size()]),
                data.rowIndicesToArray(),
                data.colIndicesToArray());
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        T transpose = makeLikeTensor(shape.swapAxes(0, 1), (W[]) FieldOps.conj(entries, null), colIndices.clone(), rowIndices.clone());
        transpose.sortIndices(); // Ensure the indices are sorted correctly.
        return transpose;
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        return T(axis1, axis2);
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public T H(int... axes) {
        return T(axes);
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public W min() {
        return (W) CompareSemiring.min(entries);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public W max() {
        return (W) CompareSemiring.max(entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        int idx = CompareSemiring.argmin(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        int idx = CompareSemiring.argmax(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        SparseMatrixData<Semiring<W>> data = CooSemiringMatrixOps.add(
                shape, entries, rowIndices, colIndices,
                b.shape, b.entries, b.rowIndices, b.colIndices);

        return makeLikeTensor(data.shape(),
                (W[]) data.entries().toArray(new Field[data.entries().size()]),
                data.rowIndicesToArray(),
                data.colIndicesToArray());
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T elemMult(T b) {
        SparseMatrixData<Semiring<W>> data = CooSemiringMatrixOps.elemMult(
                shape, entries, rowIndices, colIndices,
                b.shape, b.entries, b.rowIndices, b.colIndices);

        return makeLikeTensor(data.shape(),
                (W[]) data.entries().toArray(new Field[data.entries().size()]),
                data.rowIndicesToArray(),
                data.colIndicesToArray());
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);

        return (T) makeLikeTensor(new Shape(1, 1), (W[]) new Field[]{tr()}, new int[]{0}, new int[]{0});
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        CooDataSorter.wrap(entries, rowIndices, colIndices).sparseSort().unwrap(entries, rowIndices, colIndices);
    }


    /**
     * Converts this sparse COO matrix to an equivalent dense matrix.
     * @return A dense matrix equivalent to this sparse COO matrix.
     */
    public U toDense() {
        Field<W>[] entries = new Field[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++)
            entries[rowIndices[i]*numCols + colIndices[i]] = this.entries[i];

        return makeLikeDenseTensor(shape, entries);
    }


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    public AbstractCsrFieldMatrix<?, U, V, W> toCsr() {
        Field<W>[] csrEntries = new Field[entries.length];
        int[] csrRowPointers = new int[numRows + 1];
        int[] csrColPointers = new int[colIndices.length];
        CooConversions.toCsr(shape, entries, rowIndices, colIndices, csrEntries, csrRowPointers, csrColPointers);
        return makeLikeCsrMatrix(shape, csrEntries, csrRowPointers, csrColPointers);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public abstract AbstractCooFieldTensor<?, ?, W> toTensor();


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     * @param newShape New shape for the tensor. Can be any rank but must be broadcastable to {@link #shape this.shape}.
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    public abstract AbstractCooFieldTensor<?, ?, W> toTensor(Shape newShape);


    /**
     * Converts this sparse CSR matrix to an equivalent vector. If this matrix is not a row or column vector it will be flattened
     * before conversion.
     * @return A vector equivalent to this CSR matrix.
     */
    public V toVector() {
        int[] destIndices = new int[entries.length];
        for(int i=0; i<entries.length; i++)
            destIndices[i] = rowIndices[i]*colIndices[i];

        return makeLikeVector(new Shape(numRows*numCols), entries.clone(), destIndices);
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     * <p><b>WARNING</b>: This method is not supported for sparse tensors. If called on a sparse tensor,
     * an {@link UnsupportedOperationException} will be thrown. Element-wise division is undefined for sparse matrices as it
     * would almost certainly result in a division by zero.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     * @throws UnsupportedOperationException if this method is ever invoked on a sparse tensor.
     */
    @Override
    public T div(T b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse matrices.");
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        Field<W>[] dest = new Field[entries.length];
        FieldOps.sqrt(entries, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values. Otherwise, returns {@code false}.
     *
     * @see #isInfinite()
     * @see #isNaN()
     */
    @Override
    public boolean isFinite() {
        return FieldOps.isFinite(entries);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value. Otherwise, returns {@code false}.
     *
     * @see #isFinite()
     * @see #isNaN()
     */
    @Override
    public boolean isInfinite() {
        return FieldOps.isInfinite(entries);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value. Otherwise, returns {@code false}.
     *
     * @see #isFinite()
     * @see #isInfinite()
     */
    @Override
    public boolean isNaN() {
        return FieldOps.isNaN(entries);
    }
}
