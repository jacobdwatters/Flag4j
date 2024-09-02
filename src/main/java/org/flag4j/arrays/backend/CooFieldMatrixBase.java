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

package org.flag4j.arrays.backend;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.fields.RealFloat64;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.sparse.CooFieldMatrix;
import org.flag4j.arrays.sparse.CooFieldTensor;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.operations.dense.real.RealDenseTranspose;
import org.flag4j.operations.sparse.coo.SparseDataWrapper;
import org.flag4j.operations.sparse.coo.field_ops.*;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>A real sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
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
public abstract class CooFieldMatrixBase<T extends CooFieldMatrixBase<T, U, V, W>, U extends DenseFieldMatrixBase<U, T, ?, ?, W>,
        V extends CooFieldVectorBase<V, T, ?, U, W>, W extends Field<W>>
        extends FieldTensorBase<T, U, W>
        implements CooMatrixMixin<T, U, W> {

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
    private double sparsity = -1;


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    protected CooFieldMatrixBase(Shape shape, W[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries);
        ParameterChecks.ensureRank(shape, 2);
        ParameterChecks.ensureIndexInBounds(shape.get(0), rowIndices);
        ParameterChecks.ensureIndexInBounds(shape.get(1), colIndices);
        ParameterChecks.ensureArrayLengthsEq(entries.length, rowIndices.length, colIndices.length);

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    protected CooFieldMatrixBase(Shape shape, List<W> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape, (W[]) entries.toArray(new Field[0]));
        ParameterChecks.ensureRank(shape, 2);
        this.rowIndices = ArrayUtils.fromIntegerList(rowIndices);
        this.colIndices = ArrayUtils.fromIntegerList(colIndices);
        ParameterChecks.ensureArrayLengthsEq(super.entries.length, this.rowIndices.length, this.colIndices.length);

        nnz = super.entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public abstract T makeLikeTensor(Shape shape, W[] entries);


    /**
     * Constructs a COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Row indices of the non-zero values in the matrix.
     * @param colIndices Column indices of the non-zero values in the matrix.
     * @return A COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    public abstract T makeLikeTensor(Shape shape, W[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Constructs a COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Row indices of the non-zero values in the matrix.
     * @param colIndices Column indices of the non-zero values in the matrix.
     * @return A COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    public abstract T makeLikeTensor(Shape shape, List<W> entries, List<Integer> rowIndices, List<Integer> colIndices);


    /**
     * Constructs a dense field matrix which with the specified {@code shape} and {@code entries}.
     * @param shape Shape of the matrix.
     * @param entries Entries of the dense matrix/.
     * @return A dense field matrix with the specified {@code shape} and {@code entries}.
     */
    public abstract U makeDenseTensor(Shape shape, W[] entries);


    /**
     * Constructs a vector of similar type to this matrix.
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero values of the vector.
     * @return A vector of similar type to this matrix with the specified size, non-zero entries, and indices.
     */
    public abstract V makeLikeVector(int size, W[] entries, int[] indices);


    /**
     * Constructs a vector of similar type to this matrix.
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero values of the vector.
     * @return A vector of similar type to this matrix with the specified size, non-zero entries, and indices.
     */
    public abstract V makeLikeVector(int size, List<W> entries, List<Integer> indices);


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
     * @return The trace of this matrix. If this matrix has no non-zero entries, then {@code null} will be returned.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public W tr() {
        W trace = getZeroElement();

        for(int i=0; i<entries.length; i++)
            if(rowIndices[i]==colIndices[i]) trace = trace.add(entries[i]); // Then entry on the diagonal.

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
     *
     * @see #isCloseToI()
     */
    @Override
    public boolean isI() {
        return CooFieldMatrixProperties.isIdentity(this);
    }


    /**
     * Checks that this matrix is close to the identity matrix.
     *
     * @return True if this matrix is approximately the identity matrix.
     *
     * @see #isI()
     */
    @Override
    public boolean isCloseToI() {
        return CooFieldMatrixProperties.isCloseToIdentity(this);
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    @Override
    public W det() {
        return toDense().det();
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
        ParameterChecks.ensureMatMultShapes(shape, b.shape);

        return makeDenseTensor(new Shape(numRows, b.numCols),
                (W[]) CooFieldMatMult.standard(
                        entries, rowIndices, colIndices, shape,
                        b.entries, b.rowIndices, b.colIndices, b.shape));
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.W())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.W())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public U multTranspose(T b) {
        ParameterChecks.ensureEquals(numCols, b.numCols);
        return mult(b.T());
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix b.
     *
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    @Override
    public W fib(T b) {
        return this.H().mult(b).tr();
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
     * @see #augment(CooFieldMatrixBase) 
     */
    @Override
    public T stack(T b) {
        ParameterChecks.ensureEquals(numCols, b.numCols);

        Shape destShape = new Shape(numRows+b.numRows, numCols);
        Field<W>[] destEntries = new Field[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);

        // Copy row indices.
        int[] shiftedRowIndices = ArrayUtils.shift(numRows, b.rowIndices.clone());
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, b.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(b.colIndices, 0, destColIndices, colIndices.length, b.colIndices.length);

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
     * @see #stack(MatrixMixin, int) 
     * @see #stack(CooFieldMatrixBase) 
     */
    @Override
    public T augment(T b) {
        ParameterChecks.ensureEquals(numRows, b.numRows);

        Shape destShape = new Shape(numRows, numCols + b.numCols);
        Field<W>[] destEntries = new Field[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(b.rowIndices, 0, destRowIndices, rowIndices.length, b.rowIndices.length);

        // Copy column indices (with shifts if appropriate).
        int[] shifted = b.colIndices.clone();
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(ArrayUtils.shift(numCols, shifted), 0,
                destColIndices, colIndices.length, b.colIndices.length);

        T dest = makeLikeTensor(destShape, (W[]) destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure indices are sorted properly.

        return dest;
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
        return (T) CooFieldMatrixManipulations.swapRows(this, rowIndex1, rowIndex2);
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
        return (T) CooFieldMatrixManipulations.swapCols(this, colIndex1, colIndex2);
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     *
     * @see #isAntiSymmetric()
     */
    @Override
    public boolean isSymmetric() {
        return CooFieldMatrixProperties.isSymmetric(this);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return CooFieldMatrixProperties.isHermitian(this);
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     *
     * @see #isSymmetric()
     */
    @Override
    public boolean isAntiSymmetric() {
        return CooFieldMatrixProperties.isAntiHermitian(this);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        if(isSquare()) return this.mult(this.T()).isI();
        else return false;
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
        return (T) CooFieldMatrixManipulations.removeRow(this, rowIndex);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIndices) {
        return (T) CooFieldMatrixManipulations.removeRows(this, rowIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCol(int colIndex) {
        return (T) CooFieldMatrixManipulations.removeCol(this, colIndex);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCols(int... colIndices) {
        return (T) CooFieldMatrixManipulations.removeCols(this, colIndices);
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
        return (T) CooFieldMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
        return (T) CooFieldMatrixGetSet.getSlice(this, rowStart, rowEnd, colStart, colEnd);
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
        ParameterChecks.ensureValidIndex(shape, row, col);
        return (T) CooFieldMatrixGetSet.matrixSet(this, row, col, value);
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
        int sizeEst = nnz / 2; // Estimate the number of non-zero entries.
        List<W> triuEntries = new ArrayList<>(sizeEst);
        List<Integer> triuRowIndices = new ArrayList<>(sizeEst);
        List<Integer> triuColIndices = new ArrayList<>(sizeEst);

        for(int i=0; i<nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            if(col >= row) {
                triuEntries.add(entries[i]);
                triuRowIndices.add(row);
                triuColIndices.add(col);
            }
        }

        return makeLikeTensor(shape, triuEntries, triuRowIndices, triuColIndices);
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
        int sizeEst = nnz / 2; // Estimate the number of non-zero entries.
        List<W> trilEntries = new ArrayList<>(sizeEst);
        List<Integer> trilRowIndices = new ArrayList<>(sizeEst);
        List<Integer> trilColIndices = new ArrayList<>(sizeEst);

        for(int i=0; i<nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            if(col <= row) {
                trilEntries.add(entries[i]);
                trilRowIndices.add(row);
                trilColIndices.add(col);
            }
        }

        return makeLikeTensor(shape, trilEntries, trilRowIndices, trilColIndices);
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
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return conj();
        return H();
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
        ParameterChecks.ensureArrayLengthsEq(2, axes.length);
        ParameterChecks.ensurePermutation(axes);
        return H();
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public U tensorDot(T src2, int[] aAxes, int[] bAxes) {
        CooFieldTensor<W> t1 = new CooFieldTensor<W>(
                shape, entries, RealDenseTranspose.blockedIntMatrix(
                new int[][]{rowIndices, colIndices}));
        CooFieldTensor<W> t2 = new CooFieldTensor<W>(
                src2.shape, src2.entries, RealDenseTranspose.blockedIntMatrix(
                new int[][]{src2.rowIndices, src2.colIndices}));
        FieldMatrix<W> mat = CooFieldTensorDot.tensorDot(t1, t2, aAxes, bAxes).toMatrix();

        return makeDenseTensor(mat.shape, (W[]) mat.entries);
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
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return copy();
        return T();
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
        ParameterChecks.ensureArrayLengthsEq(2, axes.length);
        ParameterChecks.ensurePermutation(axes);
        return T();
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
     * @return The density of this sparse tensor.
     */
    @Override
    public double sparsity() {
        // Check if the sparsity has already been computed.
        if (this.sparsity < 0) {
            BigInteger totalEntries = totalEntries();
            BigDecimal sparsity = new BigDecimal(totalEntries).subtract(BigDecimal.valueOf(nnz));
            sparsity = sparsity.divide(new BigDecimal(totalEntries), RoundingMode.HALF_UP);

            this.sparsity = sparsity.doubleValue();
        }

        return sparsity;
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    @Override
    public U toDense() {
        Field<W>[] entries = new Field[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++)
            entries[rowIndices[i]*numCols + colIndices[i]] = this.entries[i];

        return makeDenseTensor(shape, (W[]) entries);
    }


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    public abstract CsrFieldMatrixBase toCsr();


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
        SparseDataWrapper.wrap(entries, rowIndices, colIndices).sparseSort().unwrap(entries, rowIndices, colIndices);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public W get(int... indices) {
        ParameterChecks.ensureEquals(indices.length, 2);
        ParameterChecks.ensureValidIndex(shape, indices[0], indices[1]);
        ParameterChecks.ensureIndexInBounds(numRows, indices[0]);
        ParameterChecks.ensureIndexInBounds(numCols, indices[1]);

        return CooFieldMatrixGetSet.matrixGet(this, indices[0], indices[1]);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        int[] destIndices = new int[entries.length];

        for(int i = 0; i < entries.length; i++)
            destIndices[i] = shape.entriesIndex(rowIndices[i], colIndices[i]);

        return makeLikeTensor(shape, entries.clone(), new int[entries.length], destIndices);
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
        ParameterChecks.ensureValidAxes(shape, axis);
        int[] dims = {1, 1};
        dims[1-axis] = this.totalEntries().intValueExact();

        int[] rowIndices = axis==1 ? this.rowIndices.clone() : new int[this.rowIndices.length];
        int[] colIndices = axis==0 ? this.colIndices.clone() : new int[this.colIndices.length];

        return makeLikeTensor(new Shape(dims), entries.clone(), rowIndices, colIndices);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        ParameterChecks.ensureBroadcastable(shape, newShape);
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

        return makeLikeTensor(newShape, entries.clone(), newRowIndices, newColIndices);
    }


    /**
     * Computes the product of all non-zero values in this tensor.
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public W prod() {
        return super.prod(); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        return (T) CooFieldMatrixOperations.add(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with the scalar {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        return (T) CooFieldMatrixOperations.sub(this, b);
    }


    /**
     * Adds a real value to each non-zero entry of this tensor.
     *
     * @param b Value to add to each non-zero value of this tensor.
     *
     * @return Sum of this tensor with {@code b}.
     */
    @Override
    public T add(double b) {
        return super.add(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Subtracts a real value from each entry of this tensor.
     *
     * @param b Value to subtract from each value of this tensor.
     *
     * @return Difference of this tensor with {@code b}.
     */
    @Override
    public T sub(double b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
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
        return (T) CooFieldMatrixOperations.elemMult(this, b);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by
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
        ParameterChecks.ensureNotEquals(axis1, axis2);
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);

        return makeLikeTensor(new Shape(1, 1), (W[]) new Field[]{tr()}, new int[]{0}, new int[]{0});
    }


    /**
     * Adds a scalar field value to each non-zero entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor's non-zero values with the scalar {@code b}.
     */
    @Override
    public T add(W b) {
        return super.add(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Adds a scalar value to each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(W b) {
        super.addEq(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Subtracts a scalar field value from each non-zero entry of this tensor.
     *
     * @param b Scalar field value in difference.
     *
     * @return The difference of this tensor's non-zero value and the scalar {@code b}.
     */
    @Override
    public T sub(W b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(W b) {
        super.subEq(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Finds the minimum non-zero value in this tensor.
     *
     * @return The minimum non-zero value in this tensor.
     */
    @Override
    public W min() {
        return super.min(); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Finds the maximum non-zero value in this tensor.
     *
     * @return The maximum non-zero value in this tensor.
     */
    @Override
    public W max() {
        return super.max(); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Finds the minimum non-zero value, in absolute value, in this tensor.
     *
     * @return The minimum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return super.minAbs(); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Finds the maximum non-zero value, in absolute value, in this tensor.
     *
     * @return The maximum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return super.maxAbs(); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Finds the indices of the minimum non-zero value in this tensor.
     *
     * @return The indices of the minimum non-zero value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        int idx = CompareField.argmin(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the maximum non-zero value in this tensor.
     *
     * @return The indices of the maximum non-zero value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        int idx = CompareField.argmax(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the minimum non-zero absolute value in this tensor.
     *
     * @return The indices of the minimum non-zero absolute value in this tensor. If this value occurs multiple times, the indices of
     * the first entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        int idx = CompareField.argminAbs(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the maximum non-zero absolute value in this tensor.
     *
     * @return The indices of the maximum non-zero absolute value in this tensor. If this value occurs multiple times, the indices of
     * the first entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        int idx = CompareField.argmaxAbs(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
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
        T transpose = makeLikeTensor(shape.swapAxes(0, 1), entries.clone(), colIndices.clone(), rowIndices.clone());
        transpose.sortIndices(); // Ensure the indices are sorted correctly.

        return transpose;
    }


    /**
     * Computes the Hermitian transpose of a tensor by exchanging and conjugating the first and last axes of this tensor.
     *
     * @return The Hermitian transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public T H() {
        T transpose = makeLikeTensor(shape.swapAxes(0, 1), entries.clone(), colIndices.clone(), rowIndices.clone());
        transpose.sortIndices(); // Ensure the indices are sorted correctly.

        return transpose.conj();
    }


    /**
     * Computes the element-wise reciprocals of the non-zero values of this tensor.
     *
     * @return A tensor containing the reciprocal non-zero elements of this tensor.
     */
    @Override
    public T recip() {
        return super.recip(); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    public V toVector() {
        int[] destIndices = new int[entries.length];
        for(int i=0; i<entries.length; i++)
            destIndices[i] = rowIndices[i]*colIndices[i];

        return makeLikeVector(numRows*numCols, entries.clone(), destIndices);
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     *
     * @return The specified row of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     *                                        the number of rows in this matrix.
     */
    public V getRow(int rowIdx) {
        return (V) CooFieldMatrixGetSet.getRow(this, rowIdx);
    }


    /**
     * Gets a specified row of this matrix between {@code colStart} (inclusive) and {@code colEnd} (exclusive).
     *
     * @param rowIdx Index of the row of this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The row at index {@code rowIdx} of this matrix between the {@code colStart} and {@code colEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code colEnd} is less than {@code colStart}.
     */
    public V getRow(int rowIdx, int colStart, int colEnd) {
        return (V) CooFieldMatrixGetSet.getRow(this, rowIdx, colStart, colEnd);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     *
     * @return The specified column of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     *                                        the number of columns in this matrix.
     */
    public V getCol(int colIdx) {
        return (V) CooFieldMatrixGetSet.getCol(this, colIdx);
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param colIdx Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     *
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     *
     * @throws @throws                  IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the
     *                                  shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    public V getCol(int colIdx, int rowStart, int rowEnd) {
        return (V) CooFieldMatrixGetSet.getCol(this, colIdx, rowStart, rowEnd);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    public V getDiag() {
        List<W> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<entries.length; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                destEntries.add(entries[i]);
                destIndices.add(rowIndices[i]);
            }
        }

        return makeLikeVector(numRows, destEntries, destIndices);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A copy of this matrix with the specified column altered.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    public T setCol(V values, int colIndex) {
        return (T) CooFieldMatrixGetSet.setCol(this, colIndex, values);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A copy of this matrix with the specified rows altered.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    public T setRow(V values, int rowIndex) {
        return (T) CooFieldMatrixGetSet.setRow(this, rowIndex, values);
    }


    /**
     * Computes the element-wise absolute value of this matrix.
     *
     * @return The element-wise absolute value of this matrix.
     */
    @Override
    public CooFieldMatrix<RealFloat64> abs() {
        RealFloat64[] abs = new RealFloat64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            abs[i] = new RealFloat64(entries[i].abs());

        return new CooFieldMatrix<RealFloat64>(shape, abs, rowIndices.clone(), colIndices.clone());
    }
}
