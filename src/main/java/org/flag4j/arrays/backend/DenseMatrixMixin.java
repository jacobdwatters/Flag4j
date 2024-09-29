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


/**
 * This interface defines operations that all dense matrices should implement.
 * @param <T> Type of this dense matrix.
 * @param <U> Type of sparse COO matrix which is equivalent to {@code T}.
 * @param <V> Type of vector which is similar to the type of this dense matrix.
 * @param <W> Type (or wrapper) of an element of this matrix.
 */
public interface DenseMatrixMixin<T extends DenseMatrixMixin<T, U, V, W>,
        U extends CooMatrixMixin<U, T, ?, V, W>,
        V extends DenseVectorMixin<V, ?, T, W>, W>
        extends MatrixMixin<T, T, V, V, W>, DenseTensorMixin<T, U> {

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
     * Sets a slice of this matrix to the specified {@code values}. The {@code rowStart} and {@code colStart} parameters specify the
     * upper left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException If the {@code values} slice, with upper left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    public T setSlice(T values, int rowStart, int colStart);


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the {@code values} array has a different shape then this matrix.
     */
    public T setValues(W[][] values);


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
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    public default boolean isFullRank() {
        return matrixRank() == Math.min(numRows(), numCols());
    }
}
