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
import org.flag4j.arrays.sparse.CsrFieldMatrix;
import org.flag4j.operations.sparse.csr.field_ops.*;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;

import static org.flag4j.operations.sparse.SparseUtils.sortCsrMatrix;


/**
 * <p>A real sparse matrix stored in compressed sparse row (CSR) format. The {@link #entries} of this CSR matrix are
 * elements of a {@link Field}.</p>
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a CSR matrix are mutable but the {@link #shape}
 * and {@link #nnz total number of non-zero entries} is fixed.</p>
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero values.</p>
 *
 * <p>A sparse CSR matrix is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowPointers row pointers} of the non-zero values in the CSR matrix. Has size {@link #numRows numRows + 1}</li>
 *     <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all values in row
 *     {@code i}.</p>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the CSR matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any operations implemented in this
 * class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.</p>
 */
public abstract class CsrFieldMatrixBase<T extends CsrFieldMatrixBase<T, U, V, W>, U extends DenseFieldMatrixBase<U, ?, T, ?, W>,
        V extends CooFieldVectorBase<V, ?, ?, U, W>, W extends Field<W>>
        extends FieldTensorBase<T, U, W>
        implements CsrMatrixMixin<T, U, W> {


    /**
     * <p>Pointers indicating starting index of each row within the {@link #colIndices} and {@link #entries} arrays.
     * Has length {@link #numRows numRows + 1}.</p>
     *
     * <p>The range {@code [entries[rowPointers[i]], entries[rowPointers[i+1]])} contains all {@link #entries non-zero entries} within
     * row {@code i}.</p>
     *
     * <p>Similarly, {@code [colIndices[rowPointers[i]], colIndices[rowPointers[i+1]])} contains all {@link #colIndices column indices}
     * for the entries in row {@code i}.
     * </p>
     */
    public final int[] rowPointers;
    /**
     * Column indices for non-zero values of this sparse CSR matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero entries in this CSR matrix.
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
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    protected CsrFieldMatrixBase(Shape shape, W[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries);
        ParameterChecks.ensureRank(shape, 2);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.nnz = entries.length;
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Constructs a CSR matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and the same non-zero
     * indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and the same non-zero
     * indices.
     */
    @Override
    public abstract T makeLikeTensor(Shape shape, W[] entries);


    /**
     * Constructs a CSR matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and non-zero
     * indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     * @param rowPointers Row pointers for the CSR matrix.
     * @param colIndices Column indices of the CSR matrix.
     *
     * @return A matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and non-zero
     * indices.
     */
    public abstract T makeLikeTensor(Shape shape, W[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Constructs a dense matrix of similar type as this matrix with the given the {@code shape} and {@code entries}.
     *
     * @param shape Shape of the dense matrix to construct.
     * @param entries Entries of the dense matrix to construct.
     *
     * @return A dense matrix of similar type as this sparse CSR matrix with the given the {@code shape} and {@code entries}.
     */
    public abstract U makeLikeDenseTensor(Shape shape, W[] entries);


    /**
     * <p>Converts a sparse COO matrix to a CSR matrix of the same type as this matrix.</p>
     * <p>The COO matrix is specified by it's shape, non-zero entries, row indices, and column indices.</p>
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero entries of the COO matrix.
     * @param rowIndices The row indices of the non-zero values in the COO matrix.
     * @param colIndices The column indices of the non-zero values in the COO matrix.
     *
     * @return A CSR matrix equivalent to the specified COO matrix.
     */
    public T fromCoo(Shape shape, List<W> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        int csrRows = shape.get(0);
        int[] csrRowPointers = new int[csrRows + 1];

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0; i<entries.size(); i++)
            csrRowPointers[rowIndices.get(i) + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<csrRows; i++)
            csrRowPointers[i+1] += csrRowPointers[i];

        return makeLikeTensor(shape, (W[]) entries.toArray(new Field[0]), csrRowPointers, ArrayUtils.fromIntegerList(colIndices));
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
        return (U) CsrFieldMatrixTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
        if(axis1 == axis2) return copy();

        return CsrFieldMatrixOperations.hermTranspose((T) this);
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
        if(axes.length != 2 || !((axes[0] == 0 && axes[1] == 1) || (axes[0] == 1 && axes[1] == 0))) {
            throw new LinearAlgebraException("Cannot transpose axes: "  + Arrays.toString(axes) + " for tensor of rank 2.");
        }

        return CsrFieldMatrixOperations.hermTranspose((T) this);
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
        ParameterChecks.ensureSquareMatrix(shape);
        W trace = getZeroElement();

        for(int i=0; i<numRows; i++) {
            int rowPtr = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowPtr; j<stop; j++)
                if(i==colIndices[j]) trace = trace.add(entries[j]);
        }

        return trace;
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CsrFieldMatrix<RealFloat64> abs() {
        RealFloat64[] absValues = new RealFloat64[nnz];
        for(int i=0; i<nnz; i++)
            absValues[i] = new RealFloat64(entries[i].abs());

        return new CsrFieldMatrix<RealFloat64>(shape, absValues, rowPointers.clone(), colIndices.clone());
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
        if(!isSquare()) return false;

        for(int i=1; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowStart; j<stop; j++) {
                if(colIndices[j] >= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(!entries[j].isZero()) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point then the matrix must be upper triangular.
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
        if(!isSquare()) return false;

        for(int i=0; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int rowStop = rowPointers[i+1];

            for(int j=rowStop-1; j>=rowStart; j--) {
                if(colIndices[j] <= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(!entries[j].isZero()) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point then the matrix must be lower-triangular.
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
        return CsrFieldMatrixProperties.isIdentity(this);
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
        return CsrFieldMatrixProperties.isCloseToIdentity(this);
    }


    /**
     * <p>Computes the determinant of a square matrix.</p>
     *
     * <p><b>Warning:</b> This method will convert this matrix to a dense matrix before computing the determinant. As such, it is
     * not advisable to call this method on large matrices.</p>
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
        return (U) CsrFieldMatMult.standard(this, b);
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
     * @see #augment(T)
     */
    @Override
    public T stack(T b) {
        return (T) toCoo().stack(b.toCoo()).toCsr();
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
        return (T) toCoo().augment(b.toCoo()).toCsr();
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
        CsrFieldMatrixManipulations.swapRows(this, rowIndex1, rowIndex2);
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
        CsrFieldMatrixManipulations.swapCols(this, colIndex1, colIndex2);
        return (T) this;
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
        return CsrFieldMatrixProperties.isSymmetric(this);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return CsrFieldMatrixProperties.isHermitian(this);
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
        return CsrFieldMatrixProperties.isAntiSymmetric(this);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        return isSquare() && this.mult(this.T()).isCloseToI();
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its hermitian transpose.
     *
     * @return True if this matrix it is unitary. Otherwise, returns false.
     */
    public boolean isUnitary() {
        return isSquare() && this.mult(this.H()).isCloseToI();
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
        return (T) toCoo().removeRow(rowIndex).toCsr();
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIndices) {
        return (T) toCoo().removeRows(rowIndices).toCsr();
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
        return (T) toCoo().removeCol(colIndex).toCsr();
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
        return (T) toCoo().removeCols(colIndices).toCsr();
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
        return (T) toCoo().setSliceCopy(values.toCoo(), rowStart, colStart).toCsr();
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
        return CsrFieldMatrixOperations.getSlice((T) this, rowStart, rowEnd, colStart, colEnd);
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
        // Ensure indices are in bounds.
        ParameterChecks.ensureValidIndex(shape, row, col);
        Field<W>[] newEntries;
        int[] newRowPointers = rowPointers.clone();
        int[] newColIndices;
        boolean found = false; // Flag indicating an element already exists in this matrix at the specified row and col.
        int loc = -1;

        if(rowPointers[row] < rowPointers[row+1]) {
            int start = rowPointers[row];
            int stop = rowPointers[row+1];

            loc = Arrays.binarySearch(colIndices, start, stop, col);
            found = loc >= 0;
        }

        if(found) {
            newEntries = entries.clone();
            newEntries[loc] = value;
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch
            newEntries = new Field[entries.length + 1];
            newColIndices = new int[entries.length + 1];

            // Copy old entries and insert new one.
            System.arraycopy(entries, 0, newEntries, 0, loc);
            newEntries[loc] = value;
            System.arraycopy(entries, loc, newEntries, loc+1, entries.length-loc);

            // Copy old column indices and insert new one.
            System.arraycopy(colIndices, 0, newColIndices, 0, loc);
            newColIndices[loc] = col;
            System.arraycopy(colIndices, loc, newColIndices, loc+1, entries.length-loc);

            // Increment row pointers.
            for(int i=row+1; i<rowPointers.length; i++) {
                newRowPointers[i]++;
            }
        }

        return makeLikeTensor(shape, (W[]) newEntries, newRowPointers, newColIndices);
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
        return (T) toCoo().getTriU(diagOffset).toCsr();
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
        return (T) toCoo().getTriL(diagOffset).toCsr();
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

        return CsrFieldMatrixOperations.transpose((T) this);
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
        if(axes.length != 2 || !((axes[0] == 0 && axes[1] == 1) || (axes[0] == 1 && axes[1] == 0))) {
            throw new LinearAlgebraException("Cannot transpose axes: "  + Arrays.toString(axes) + " for tensor of rank 2.");
        }

        return CsrFieldMatrixOperations.transpose((T) this);
    }


    /**
     * The sparsity of this sparse CSR matrix. That is, the decimal percentage of elements in this matrix which are zero.
     *
     * @return The density of this sparse matrix.
     */
    @Override
    public double sparsity() {
        // Compute sparsity if needed.
        if(this.sparsity == -1) {
            BigDecimal sparsity = new BigDecimal(this.totalEntries()).subtract(BigDecimal.valueOf(this.nnz));
            sparsity = sparsity.divide(new BigDecimal(this.totalEntries()), 50, RoundingMode.HALF_UP);
            this.sparsity = sparsity.doubleValue();
        }

        return sparsity;
    }


    /**
     * Converts this sparse matrix to an equivalent dense matrix.
     *
     * @return A dense matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public U toDense() {
        Field<W>[] dest = new Field[shape.totalEntriesIntValueExact()];

        for(int i=0; i<rowPointers.length-1; i++) {
            int rowOffset = i*numCols;

            for(int j=rowPointers[i], rowEnd=rowPointers[i+1]; j<rowEnd; j++) {
                dest[rowOffset + colIndices[j]] = entries[j];
            }
        }

        return makeLikeDenseTensor(shape, (W[]) dest);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    public abstract CooFieldMatrixBase toCoo();


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
        sortCsrMatrix(entries, rowPointers, colIndices);
    }
}
