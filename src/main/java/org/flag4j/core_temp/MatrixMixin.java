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

package org.flag4j.core_temp;


import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.util.ErrorMessages;

/**
 * This interface defines operations that all matrices should implement.
 * @param <T> Type of this matrix.
 * @param <U> Tensor super-type of this matrix.
 * @param <V> Type (or wrapper) of an element of this matrix.
 */
public interface MatrixMixin<T extends U, U extends TensorBase<U, ?, V>, V>
        extends TensorBinaryOpsMixin<U, U>,
        TensorUnaryOpsMixin<U>,
        TensorPropertiesMixin<V> {

    // TODO: Create seperate interface MatrixVectorOps mixin for operations between a matrix and a vector.

    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    public int numRows();


    /**
     * Gets the number of columns in this matrix.
     * @return The number of columns in this matrix.
     */
    public int numCols();


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.</p>
     *
     * <p>Same as {@link #tr()}.</p>
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    public default Double trace() {return tr();}


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.</p>
     *
     * <p>Same as {@link #trace()}.</p>
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    public Double tr();


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    public default boolean isSquare() {
        return numRows()==numCols();
    }


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     *
     * @return True if this matrix can be represented as either a row or column vector.
     */
    public default boolean isVector() {
        return numRows()<=1 || numCols()<=1;
    }


    /**
     * Checks what type of vector this matrix is. i.e. not a vector, a 1x1 matrix, a row vector, or a column vector.
     *
     * @return An int corresponding to the type of vector this matrix represents:
     * <ul>
     *     <li>If this matrix can not be represented as a vector, then returns -1.</li>
     *     <li>If this matrix is a 1x1 matrix, then returns 0.</li>
     *     <li>If this matrix is a row vector, then returns 1.</li>
     *     <li>If this matrix is a column vector, then returns 2.</li>
     * </ul>
     */
    public default int vectorType() {
        int type;

        int rows = numRows();
        int cols = numCols();

        if(rows==1 || cols==1) {
            if(rows==1 && cols==1) {
                type = 0;
            } else if(rows==1) {
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
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     * @return True is this matrix is triangular. Otherwise, returns false.
     */
    default boolean isTri() {
        return isTriL() || isTriU();
    }


    /**
     * Checks if this matrix is diagonal.
     * @return True is this matrix is diagonal. Otherwise, returns false.
     */
    default boolean isDiag() {
        return isTriL() && isTriU();
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    public boolean isTriU();


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    public boolean isTriL();


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    public default boolean isFullRank() {
        return matrixRank() == Math.min(numRows(), numCols());
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.
     *
     * @return True if this matrix is singular or non-square. Otherwise, returns false.
     * @see #isInvertible()
     */
    public boolean isSingular();


    /**
     * Checks if a matrix is invertible.
     * @return True if this matrix is invertible.
     * @see #isSingular()
     */
    default boolean isInvertible() {
        return !isSingular();
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If this matrix is not square.
     */
    public V det();


    /**
     * <p>Computes the rank of this matrix (i.e. the number of linearly independent rows/columns in this matrix).</p>
     *
     * <p>Note that here, rank is <b>NOT</b> the same as a tensor rank (i.e. number of indices needed to specify an entry in
     * the tensor).</p>
     *
     * @return The matrix rank of this matrix.
     */
    public int matrixRank();


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix b.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If the number of columns in this matrix do not equal the number of rows in matrix b.
     */
    public T mult(T b);


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    public T multTranspose(T b);


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix b.
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    public V fib(T b);


    /**
     * The transpose of this matrix.
     * @return The transpose of this matrix.
     */
    @Override
    public T T();


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(T, int)
     * @see #augment(T)
     */
    public T stack(T b);


    /**
     * Stacks matrices along rows.
     *
     * @param b MatrixOld to stack to this matrix.
     * @return The result of stacking {@code b} to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(T)
     * @see #stack(T, int)
     */
    public T augment(T b);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(T)} and {@link #augment(T)}.
     *
     * @param b Matrix to stack to this matrix.
     * @param axis Axis along which to stack:
     *      <ul>
     *          <li>If axis=0, then stacks along rows and is equivalent to {@link #augment(T)}.</li>
     *          <li>If axis=1, then stacks along columns and is equivalent to {@link #stack(T)}.</li>
     *      </ul>
     *
     * @return The result of stacking this matrix and {@code b} along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     * @see #augment(T)
     * @see #stack(T)
     */
    default T stack(T b, int axis){
        if(axis == 0)
            return this.augment(b);
        else if(axis == 1)
            return this.stack(b);
        else
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
    }


    /**
     * Swaps specified rows in the matrix. This is done in place.
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     * @return A reference to this matrix.
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    public T swapRows(int rowIndex1, int rowIndex2);


    /**
     * Swaps specified columns in the matrix. This is done in place.
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     * @return A reference to this matrix.
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    public T swapCols(int colIndex1, int colIndex2);


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     * @return True if this matrix is symmetric. Otherwise, returns false.
     * @see #isAntiSymmetric()
     */
    public boolean isSymmetric();


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     * @see #isSymmetric()
     */
    public boolean isAntiSymmetric();


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    public boolean isOrthogonal();


    /**
     * Removes a specified row from this matrix.
     * @param rowIndex Index of the row to remove from this matrix.
     * @return A copy of this matrix with the specified column removed.
     */
    public T removeRow(int rowIndex);


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     * @return a copy of this matrix with the specified column removed.
     */
    public T removeRows(int... rowIndices);


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    public T removeCol(int colIndex);


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     * @return a copy of this matrix with the specified column removed.
     */
    public T removeCols(int... colIndices);


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
    public T setSliceCopy(T values, int rowStart, int colStart);


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
    public T setSlice(T values, int rowStart, int colStart);


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
    public MatrixOld getSlice(int rowStart, int rowEnd, int colStart, int colEnd);


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A reference to this matrix.
     */
    public T set(V value, int row, int col);
}
