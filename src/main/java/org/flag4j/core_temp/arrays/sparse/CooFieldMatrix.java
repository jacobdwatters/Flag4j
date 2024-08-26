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

package org.flag4j.core_temp.arrays.sparse;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.FieldTensorBase;
import org.flag4j.core_temp.MatrixMixin;
import org.flag4j.core_temp.TensorBase;
import org.flag4j.core_temp.arrays.dense.FieldMatrix;
import org.flag4j.core_temp.structures.fields.Field;
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

// TODO: Need to override some methods from FieldTensorBase to ensure they are valid for a sparse tensor.
/**
 * <p>A real sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * elements of a {@link Field}.</p>
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero vlues.</p>
 *
 * <p>COO matrices are optimized for hyper-sparse matrices (i.e. matrices which contain almost all zeros relative to the size of the
 * matrix).</p>
 *
 * <p>A sparse COO matrix is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicity be stored in {@link #entries}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matirx.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matirx.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicity verified but any operations implemented in this
 * class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.</p>
 */
public class CooFieldMatrix<T extends Field<T>> extends FieldTensorBase<CooFieldMatrix<T>, FieldMatrix<T>, T>
        implements SparseTesnorMixin<FieldMatrix<T>, CooFieldMatrix<T>>,
        MatrixMixin<CooFieldMatrix<T>, FieldMatrix<T>, T> {

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
     * @param colIndices Non-zero column indies of this sparse mattrix.
     */
    public CooFieldMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries);
        ParameterChecks.ensureRank(2, shape);
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
     * @param colIndices Non-zero column indies of this sparse mattrix.
     */
    public CooFieldMatrix(Shape shape, List<T> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape, (T[]) entries.toArray(new Field[0]));
        ParameterChecks.ensureRank(2, shape);
        this.rowIndices = ArrayUtils.fromIntegerList(rowIndices);
        this.colIndices = ArrayUtils.fromIntegerList(colIndices);
        ParameterChecks.ensureArrayLengthsEq(super.entries.length, this.rowIndices.length, this.colIndices.length);

        nnz = super.entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
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
    public T tr() {
        return null;
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
        return false;
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
        return false;
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
        return false;
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
        return false;
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    @Override
    public T det() {
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
    public FieldMatrix<T> mult(CooFieldMatrix<T> b) {
        ParameterChecks.ensureMatMultShapes(shape, b.shape);

        return new FieldMatrix<T>(numRows, b.numCols,
                (T[]) CooFieldMatMult.standard(
                        entries, rowIndices, colIndices, shape,
                        b.entries, b.rowIndices, b.colIndices, b.shape));
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
    public FieldMatrix<T> multTranspose(CooFieldMatrix<T> b) {
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
    public T fib(CooFieldMatrix<T> b) {
        return this.T().mult(b).tr();
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(TensorBase, int)
     * @see #augment(CooFieldMatrix)
     */
    @Override
    public CooFieldMatrix<T> stack(CooFieldMatrix<T> b) {
        ParameterChecks.ensureEquals(numCols, b.numCols);

        Shape destShape = new Shape(numRows+b.numRows, numCols);
        Field<T>[] destEntries = new Field[entries.length + b.entries.length];
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

        return new CooFieldMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(TensorBase, int)
     * @see #stack(CooFieldMatrix)
     */
    @Override
    public CooFieldMatrix<T> augment(CooFieldMatrix<T> b) {
        ParameterChecks.ensureEquals(numRows, b.numRows);

        Shape destShape = new Shape(numRows, numCols + b.numCols);
        Field<T>[] destEntries = new Field[entries.length + b.entries.length];
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

        CooFieldMatrix<T> dest = new CooFieldMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooFieldMatrix<T> swapRows(int rowIndex1, int rowIndex2) {
        return CooFieldMatrixManipulations.swapRows(this, rowIndex1, rowIndex2);
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
    public CooFieldMatrix<T> swapCols(int colIndex1, int colIndex2) {
        return CooFieldMatrixManipulations.swapCols(this, colIndex1, colIndex2);
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
     * Checks if a marix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
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
    public CooFieldMatrix<T> removeRow(int rowIndex) {
        return CooFieldMatrixManipulations.removeRow(this, rowIndex);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooFieldMatrix<T> removeRows(int... rowIndices) {
        return CooFieldMatrixManipulations.removeRows(this, rowIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooFieldMatrix<T> removeCol(int colIndex) {
        return CooFieldMatrixManipulations.removeCol(this, colIndex);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooFieldMatrix<T> removeCols(int... colIndices) {
        return CooFieldMatrixManipulations.removeCols(this, colIndices);
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
    public CooFieldMatrix<T> setSliceCopy(CooFieldMatrix<T> values, int rowStart, int colStart) {
        return CooFieldMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooFieldMatrix<T> getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        return CooFieldMatrixGetSet.getSlice(this, rowStart, rowEnd, colStart, colEnd);
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
    public CooFieldMatrix<T> set(T value, int row, int col) {
        ParameterChecks.ensureValidIndex(shape, row, col);
        return CooFieldMatrixGetSet.matrixSet(this, row, col, value);
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
    public CooFieldMatrix<T> getTriU(int diagOffset) {
        // TODO: Coould probably implement a generic form of this method and use it here and in the real CooMatrix class.
        int sizeEst = nnz / 2; // Esitimate the number of non-zero entries.
        List<T> triuEntries = new ArrayList<>(sizeEst);
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

        return new CooFieldMatrix(shape, triuEntries, triuRowIndices, triuColIndices);
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
    public CooFieldMatrix<T> getTriL(int diagOffset) {
        int sizeEst = nnz / 2; // Esitimate the number of non-zero entries.
        List<T> trilEntries = new ArrayList<>(sizeEst);
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

        return new CooFieldMatrix(shape, trilEntries, trilRowIndices, trilColIndices);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public CooFieldMatrix<T> H(int axis1, int axis2) {
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
    public CooFieldMatrix<T> H(int... axes) {
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
    public FieldMatrix<T> tensorDot(CooFieldMatrix<T> src2, int[] aAxes, int[] bAxes) {
        CooFieldTensor<T> t1 = new CooFieldTensor<T>(
                shape, entries, RealDenseTranspose.blockedIntMatrix(
                        new int[][]{rowIndices, colIndices}));
        CooFieldTensor<T> t2 = new CooFieldTensor<T>(
                src2.shape, src2.entries, RealDenseTranspose.blockedIntMatrix(
                        new int[][]{src2.rowIndices, src2.colIndices}));

        return CooFieldTensorDot.tensorDot(t1, t2, aAxes, bAxes).toMatrix();
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entires of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public CooFieldMatrix<T> T(int axis1, int axis2) {
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
    public CooFieldMatrix<T> T(int... axes) {
        ParameterChecks.ensureArrayLengthsEq(2, axes.length);
        ParameterChecks.ensurePermutation(axes);
        return T();
    }


    /**
     * The sparsity of this sparse tensor. That is, the precentage of elements in this tensor which are zero as a decimal.
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
     * Converts this sparse tesnor to an equivalent dense tensor.
     *
     * @return A dense tesnor equivalent to this sparse tensor.
     */
    @Override
    public FieldMatrix<T> toDense() {
        Field<T>[] entries = new Field[totalEntries().intValueExact()];
        int row;
        int col;

        for(int i = 0; i< nnz; i++) {
            row = rowIndices[i];
            col = colIndices[i];
            entries[row*numCols + col] = this.entries[i];
        }

        return new FieldMatrix(shape, entries);
    }


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
    public T get(int... indices) {
        // TODO: Implementation.
        return super.get(indices);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CooFieldMatrix<T> flatten() {
        // TODO: Implementation.
        return super.flatten();
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
    public CooFieldMatrix<T> flatten(int axis) {
        // TODO: Implementation.
        return super.flatten(axis);
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
    public CooFieldMatrix<T> reshape(Shape newShape) {
        // TODO: Implementation.
        return super.reshape(newShape);
    }


    /**
     * Computes the product of all values in this tensor.
     *
     * @return The product of all values in this tensor.
     */
    @Override
    public T prod() {
        // TODO: Implementation.
        return super.prod();
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
    public CooFieldMatrix<T> add(CooFieldMatrix<T> b) {
        // TODO: Implementation.
        return super.add(b);
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
    public CooFieldMatrix<T> sub(CooFieldMatrix<T> b) {
        // TODO: Implementation.
        return super.sub(b);
    }


    /**
     * Adds a real value to each entry of this tensor.
     *
     * @param b Value to add to each value of this tensor.
     *
     * @return Sum of this tensor with {@code b}.
     */
    @Override
    public CooFieldMatrix<T> add(double b) {
        // TODO: Implementation.
        return super.add(b);
    }


    /**
     * Subtracts a real value from each entry of this tensor.
     *
     * @param b Value to subtract from each value of this tensor.
     *
     * @return Difference of this tensor with {@code b}.
     */
    @Override
    public CooFieldMatrix<T> sub(double b) {
        // TODO: Implementation.
        return super.sub(b);
    }


    /**
     * Multiplies a real value to each entry of this tensor.
     *
     * @param b Value to multiply to each value of this tensor.
     *
     * @return Product of this tensor with {@code b}.
     */
    @Override
    public CooFieldMatrix<T> mult(double b) {
        // TODO: Implementation.
        return super.mult(b);
    }


    /**
     * Divieds each entry of this tensor by a real value.
     *
     * @param b Value to divied each value of this tensor by.
     *
     * @return Quotient of this tensor with {@code b}.
     */
    @Override
    public CooFieldMatrix<T> div(double b) {
        // TODO: Implementation.
        return super.div(b);
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
    public CooFieldMatrix<T> elemMult(CooFieldMatrix<T> b) {
        // TODO: Implementation.
        return super.elemMult(b);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specifieed by
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
     *                                   (i.e. the axes are equal or the tesnor does not have the same length along the two axes.)
     */
    @Override
    public CooFieldMatrix<T> tensorTr(int axis1, int axis2) {
        // TODO: Implementation.
        return super.tensorTr(axis1, axis2);
    }


    /**
     * Adds a sclar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public CooFieldMatrix<T> add(T b) {
        // TODO: Implementation.
        return super.add(b);
    }


    /**
     * Adds a sclar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(T b) {
        // TODO: Implementation.
        super.addEq(b);
    }


    /**
     * Subtracts a sclar field value from each entry of this tensor.
     *
     * @param b Scalar field value in differencce.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public CooFieldMatrix<T> sub(T b) {
        // TODO: Implementation.
        return super.sub(b);
    }


    /**
     * Subtracts a sclar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in differencce.
     */
    @Override
    public void subEq(T b) {
        // TODO: Implementation.
        super.subEq(b);
    }


    /**
     * Multiplies a sclar field value to each entry of this tensor.
     *
     * @param b Scalar field value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public CooFieldMatrix<T> mult(T b) {
        // TODO: Implementation.
        return super.mult(b);
    }


    /**
     * Multiplies a sclar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(T b) {
        // TODO: Implementation.
        super.multEq(b);
    }


    /**
     * Divides each entry of this tensor by a scalar field element.
     *
     * @param b Scalar field value in quotient.
     *
     * @return The quotient of this tensor with {@code b}.
     */
    @Override
    public CooFieldMatrix<T> div(T b) {
        // TODO: Implementation.
        return super.div(b);
    }


    /**
     * Divides each entry of this tensor by s scalar field element and stores the result in this tensor.
     *
     * @param b Scalar field value in quotient.
     */
    @Override
    public void divEq(T b) {
        // TODO: Implementation.
        super.divEq(b);
    }


    /**
     * Finds the minimum value in this tensor.
     *
     * @return The minimum value in this tensor.
     */
    @Override
    public T min() {
        // TODO: Implementation.
        return super.min();
    }


    /**
     * Finds the maximum value in this tensor.
     *
     * @return The maximum value in this tensor.
     */
    @Override
    public T max() {
        // TODO: Implementation.
        return super.max();
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        // TODO: Implementation.
        return super.minAbs();
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        // TODO: Implementation.
        return super.maxAbs();
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        // TODO: Implementation.
        return super.argmin();
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        // TODO: Implementation.
        return super.argmax();
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        // TODO: Implementation.
        return super.argminAbs();
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        // TODO: Implementation.
        return super.argmaxAbs();
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
    public CooFieldMatrix<T> T() {
        CooFieldMatrix<T> transpose = new CooFieldMatrix<T>(
                shape.swapAxes(0, 1),
                entries.clone(),
                colIndices.clone(),
                rowIndices.clone()
        );

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
    public CooFieldMatrix<T> H() {
        CooFieldMatrix<T> transpose = new CooFieldMatrix<T>(
                shape.swapAxes(0, 1),
                entries.clone(),
                colIndices.clone(),
                rowIndices.clone()
        );

        transpose.sortIndices(); // Ensure the indices are sorted correctly.

        return transpose.conj();
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     */
    @Override
    public CooFieldMatrix<T> recip() {
        // TODO: Implementation.
        return super.recip();
    }
}
