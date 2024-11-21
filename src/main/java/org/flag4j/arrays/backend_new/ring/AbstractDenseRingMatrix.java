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

package org.flag4j.arrays.backend_new.ring;


import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.MatrixMixin;
import org.flag4j.arrays.backend_new.SparseMatrixData;
import org.flag4j.linalg.operations.TransposeDispatcher;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiRingMatMultDispatcher;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;

/**
 * The base class for all dense matrices whose elements are members of a {@link Ring}.
 * @param <T> The type of this matrix.
 * @param <U> The type of the vector which is of similar type to {@link T &lt;T&gt;}.
 * @param <V> The type of the ring the entries of the matrix belong to.
 */
public abstract class AbstractDenseRingMatrix<T extends AbstractDenseRingMatrix<T, U, V>,
        U extends AbstractDenseRingVector<U, T, V>, V extends Ring<V>>
        extends AbstractDenseRingTensor<T, V> implements MatrixMixin<T, T, U, V> {

    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractDenseRingMatrix(Shape shape, Ring<V>[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     * @param shape Shape of the vector to construct. Must be rank 1.
     * @param entries Entries of the vector.
     * @return A vector of a similar type as this matrix.
     */
    protected abstract U makeLikeVector(Shape shape, Ring<V>[] entries);


    /**
     * Constructs a sparse COO matrix which is of a similar type as this dense matrix.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @return A sparse COO matrix which is of a similar type as this dense matrix.
     */
    protected abstract AbstractCooRingMatrix<?, T, ?, V> makeLikeCooMatrix(
            Shape shape, Ring<V>[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Constructs a sparse CSR matrix which is of a similar type as this dense matrix.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A sparse CSR matrix which is of a similar type as this dense matrix.
     */
    protected abstract AbstractCsrRingMatrix<?, T, ?, V> makeLikeCsrMatrix(
            Shape shape, Ring<V>[] entries, int[] rowPointers, int[] colIndices);


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
        Ring<V>[] dest = new Ring[entries.length];
        TransposeDispatcher.dispatch(entries, shape, dest);
        return makeLikeTensor(shape.swapAxes(0, 1), dest);
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
    public V tr() {
        ValidateParameters.ensureSquareMatrix(shape);
        Ring<V> sum = entries[0];
        int colsOffset = this.numCols + 1;

        for(int i=1; i<numRows; i++)
            sum = sum.add((V) entries[i*colsOffset]);

        return (V) sum;
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

        // Ensure lower half is zeros.
        for(int i=1; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<i; j++)
                if(!entries[rowOffset + j].isOne()) return false; // No need to continue.
        }

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
        if(!isSquare()) return false;

        // Ensure upper half is zeros.
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=i+1; j<numCols; j++)
                if(!entries[rowOffset + j].isOne()) return false; // No need to continue.
        }

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
        int pos = 0;

        if(isSquare()) {
            for(int i=0; i<numRows; i++) {
                for(int j=0; j<numCols; j++) {
                    if((i==j && !entries[pos].isOne()) || i!=j && !entries[pos].isZero()) {
                        return false; // No need to continue
                    }

                    pos++;
                }
            }
        } else {
            return false; // An identity matrix must be square.
        }

        // If we make it to this point this matrix must be an identity matrix.
        return true;
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
    public T mult(T b) {
        Ring<V>[] dest = new Ring[numRows*b.numCols];
        DenseSemiRingMatMultDispatcher.dispatch(entries, shape, b.entries, b.shape, dest);
        return makeLikeTensor(new Shape(numRows, b.numCols), dest);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method <i>may</i>, be noticeably faster than directly computing the transpose followed by the
     * multiplication as {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public T multTranspose(T b) {
        Ring<V>[] dest = new Ring[numRows*b.numRows];
        DenseSemiRingMatMultDispatcher.dispatchTranspose(entries, shape, b.entries, b.shape, dest);
        return makeLikeTensor(new Shape(numRows, b.numRows), dest);
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
        ValidateParameters.ensureArrayLengthsEq(this.numCols, b.numCols);
        Shape stackedShape = new Shape(this.numRows + b.numRows, this.numCols);
        Ring<V>[] stackedEntries = new Ring[stackedShape.totalEntries().intValueExact()];

        System.arraycopy(this.entries, 0, stackedEntries, 0, this.entries.length);
        System.arraycopy(b.entries, 0, stackedEntries, this.entries.length, b.entries.length);

        return makeLikeTensor(stackedShape, stackedEntries);
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
        ValidateParameters.ensureArrayLengthsEq(numRows, b.numRows);

        int augNumCols = numCols + b.numCols;
        Shape augShape = new Shape(numRows, augNumCols);
        Ring<V>[] augEntries = new Ring[numRows*augNumCols];

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augEntries, i*augNumCols, numCols);
            int augOffset = i*augNumCols + numCols;
            int bOffset = i*b.numCols;

            for(int j=0; j<b.numCols; j++)
                augEntries[augOffset + j] = b.entries[bOffset + j];
        }

        return makeLikeTensor(augShape, augEntries);
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public T augment(U b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.size);
        Ring<V>[] augmented = new Ring[numRows*(numCols + 1)];

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augmented, i*(numCols+1), numCols);
            augmented[i*(numCols+1) + numCols] = b.entries[i];
        }

        return makeLikeTensor(new Shape(numRows, numCols+1), augmented);
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
        ValidateParameters.ensureValidArrayIndices(numRows, rowIndex1, rowIndex2);

        int row1Offset = rowIndex1*numCols;
        int row2Offset = rowIndex2*numCols;

        if(rowIndex1 != rowIndex2) {
            Ring<V> temp;

            for(int j=0; j<numCols; j++) {
                // Swap elements.
                temp = entries[row1Offset + j];
                entries[row1Offset + j] = entries[row2Offset + j];
                entries[row2Offset + j] = temp;
            }
        }

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
        ValidateParameters.ensureValidArrayIndices(numCols, colIndex1, colIndex2);

        if(colIndex1 != colIndex2) {
            Ring<V> temp;

            for(int i=0; i<numRows; i++) {
                // Swap elements.
                int idx = i*numCols;
                ArrayUtils.swap(entries, idx + colIndex1, idx + colIndex2);
            }
        }

        return (T) this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        if(this==null) return false;
        if(this.entries.length==0) return true;

        return numRows==numCols && this.equals(this.T());
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        if(this==null) return false;
        if(this.entries.length==0) return true;

        return numRows==numCols && this.equals(this.H());
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is approximately equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        return numRows == numCols && mult(T()).isI();
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
        Shape copyShape = new Shape(numRows-1, numCols);
        Ring<V>[] copyEntries = new Ring[(numRows-1)*numCols];
        int row = 0;

        for(int i=0; i<numRows; i++) {
            if(i!=rowIndex) {
                System.arraycopy(entries, i*numCols, copyEntries, row*numCols, numCols);
                row++;
            }
        }

        return makeLikeTensor(new Shape(numRows-1, numCols), copyEntries);
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
        Shape copyShape = new Shape(numRows-rowIndices.length, numCols);
        Ring<V>[] copyEntries = new Ring[(numRows-rowIndices.length)*numCols];
        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(ArrayUtils.notContains(rowIndices, i)) {
                System.arraycopy(this.entries, i*numCols, copyEntries, row*numCols, numCols);
                row++;
            }
        }

        return makeLikeTensor(new Shape(numRows-rowIndices.length, numCols), copyEntries);
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
        int copyNumCols = numCols-1;
        Shape copyShape = new Shape(numRows, copyNumCols);
        Ring<V>[] copyEntries = new Ring[numRows*copyNumCols];

        for(int i=0; i<this.numRows; i++) {
            int rowOffset = i*numCols;
            int copyOffset = i*copyNumCols;
            int  col = 0;

            for(int j=0; j<this.numCols; j++) {
                if(j!=colIndex) {
                    copyEntries[copyOffset + col] = this.entries[rowOffset + j];
                    col++;
                }
            }
        }

        return makeLikeTensor(new Shape(numRows, numCols-1), copyEntries);
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
        int copyNumCols = this.numCols-colIndices.length;
        Shape copyShape = new Shape(numRows, copyNumCols);
        Ring<V>[] copyEntries = new Ring[numRows*copyNumCols];

        for(int i=0; i<this.numRows; i++) {
            int rowOffset = i*numCols;
            int copyOffset = i*copyNumCols;
            int col = 0;

            for(int j=0; j<this.numCols; j++) {
                if(ArrayUtils.notContains(colIndices, j)) {
                    copyEntries[copyOffset + col] = this.entries[rowOffset + j];
                    col++;
                }
            }
        }

        return makeLikeTensor(new Shape(numRows, numCols-colIndices.length), copyEntries);
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    public T setSlice(T values, int rowStart, int colStart) {
        ValidateParameters.ensureValidArrayIndices(numRows, rowStart);
        ValidateParameters.ensureValidArrayIndices(numCols, colStart);

        for(int i=0; i<values.numRows; i++) {
            int src1Offset = (i+rowStart)*numCols + colStart;
            int src2RowOffset = i*values.numCols;

            for(int j=0; j<values.numCols; j++) {
                this.entries[src1Offset + j] = values.entries[src2RowOffset + j];
            }
        }

        return (T) this;
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
        return copy().setSlice(values, rowStart, colStart);
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
        ValidateParameters.ensureValidArrayIndices(numRows, rowStart, rowEnd);
        ValidateParameters.ensureValidArrayIndices(numCols, colStart, colEnd);

        int sliceRows = rowEnd-rowStart;
        int sliceCols = colEnd-colStart;
        int destPos = 0;
        Ring<V>[] slice = new Ring[sliceRows*sliceCols];

        for(int i=rowStart; i<rowEnd; i++) {
            int srcPos = i*numCols + colStart;
            int end = srcPos + colEnd - colStart;

            while(srcPos < end)
                slice[destPos++] = entries[srcPos++];
        }

        return makeLikeTensor(new Shape(sliceRows, sliceCols), slice);
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
    public T set(V value, int row, int col) {
        return super.set(value, row, col);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    public T setValues(V[][] values) {
        ValidateParameters.ensureEquals(numRows, values.length);
        ValidateParameters.ensureEquals(numCols, values[0].length);

        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<numCols; j++)
                entries[rowOffset + j] = values[i][j];
        }

        return (T) this;
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
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        Ring<V>[] copyEntries = new Ring[entries.length];
        Arrays.fill(copyEntries, (entries.length > 0) ? entries[0].getZero() : null);
        T result = makeLikeTensor(shape, copyEntries);

        // Extract the upper triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=Math.max(0, i + diagOffset); j<numCols; j++) {
                if (j >= i + diagOffset)
                    result.entries[rowOffset + j] = entries[rowOffset + j];
            }
        }

        return result;
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
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        Ring<V>[] copyEntries = new Ring[entries.length];
        Arrays.fill(copyEntries, (entries.length > 0) ? entries[0].getZero() : null);
        T result = makeLikeTensor(shape, copyEntries);

        // Extract the lower triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j <= Math.min(numCols - 1, i + diagOffset); j++) {
                if(j <= i + diagOffset) {
                    result.entries[rowOffset + j] = this.entries[rowOffset + j];
                }
            }
        }

        return result;
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
    public U getDiag(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -(numRows-1), numCols-1, "diagOffset");

        // Check for some quick returns.
        if(numRows == 1 && diagOffset > 0) return (U) makeLikeVector(shape, new Ring[]{entries[diagOffset]});
        if(numCols == 1 && diagOffset < 0) return (U) makeLikeVector(shape, new Ring[]{entries[-diagOffset]});

        // Compute the length of the diagonal.
        int newSize = Math.min(numRows, numCols);
        int idx = 0;

        if(diagOffset > 0) {
            newSize = Math.min(newSize, numCols - diagOffset);
            idx = diagOffset;
        }
        else if(diagOffset < 0) {
            newSize = Math.min(newSize, numRows + diagOffset);
            idx = -diagOffset*numCols;
        }

        Ring<V>[] diag = new Ring[newSize];

        for(int i=0; i<newSize; i++) {
            diag[i] = this.entries[idx];
            idx += numCols + 1;
        }

        return makeLikeVector(shape, diag);
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
    @Override
    public U getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.ensureIndexInBounds(numCols, colStart, colEnd-1);
        ValidateParameters.ensureGreaterEq(colStart, colEnd);
        int start = rowIdx*numCols + colStart;
        int stop = rowIdx*numCols + colEnd;

        Ring<V>[] row = Arrays.copyOfRange(this.entries, start, stop);

        return makeLikeVector(shape, row);
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
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the  shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public U getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.ensureValidArrayIndices(numRows, rowStart, rowEnd);
        ValidateParameters.ensureGreaterEq(rowEnd, rowStart);
        Ring<V>[] col = new Ring[numRows];

        for(int i=rowStart; i<rowEnd; i++)
            col[i] = entries[i*numCols + colIdx];

        return makeLikeVector(shape, col);
    }


    /**
     * Flattens this matrix to a row vector.
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
     * Flattens this matrix along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @return If {@code axis == 0} a matrix with the shape {@code (1, this.numRows*this.numCols)} is returned.
     * If {@code axis == 1} a matrix with the shape {@code (this.numRows*this.numCols, 1)} is returned.
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        return (axis == 0)
                ? makeLikeTensor(new Shape(1, entries.length), entries.clone())
                : makeLikeTensor(new Shape(entries.length, 1), entries.clone());
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        return T(); // Conjugation is not defined on a general semi-ring. Fall back to standard transpose.
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     * @see #toCoo(double)
     */
    public AbstractCooRingMatrix<?, T, ?, V> toCoo() {
        return toCoo(0.01);
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     * @see #toCoo()
     */
    public AbstractCooRingMatrix<?, T, ?, V> toCoo(double estimatedSparsity) {
        SparseMatrixData<Semiring<V>> data = DenseSemiringConversions.toCoo(shape, entries, 0.1);
        Ring<V>[] cooEntries = data.entries().toArray(new Ring[data.entries().size()]);
        int[] rowIndices = ArrayUtils.fromIntegerList(data.rowIndices());
        int[] colIndices = ArrayUtils.fromIntegerList(data.colIndices());

        return makeLikeCooMatrix(data.shape(), cooEntries, rowIndices, colIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     * @see #toCsr(double)
     */
    public AbstractCsrRingMatrix<?, T, ?, V> toCsr() {
        return toCoo(0.01).toCsr();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     * @see #toCsr()
     */
    public AbstractCsrRingMatrix<?, T, ?, V> toCsr(double estimatedSparsity) {
        return toCoo(estimatedSparsity).toCsr();
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not a row or column vector it will first be flattened then
     * converted to a vector.
     *
     * @return A vector which contains the same entries as this matrix.
     */
    @Override
    public U toVector() {
        return makeLikeVector(new Shape(numRows*numCols), entries.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor.
     * @return A tensor with the same shape and entries as this matrix.
     */
    public abstract AbstractDenseRingTensor<?, V> toTensor();


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     * @param newShape Shape of the tensor. Can be any rank but must be broadcastable to the shape of this matrix.
     * @return A tensor with the specified {@code newShape} and the same entries as this matrix.
     */
    public abstract AbstractDenseRingTensor<?, V> toTensor(Shape newShape);
}
