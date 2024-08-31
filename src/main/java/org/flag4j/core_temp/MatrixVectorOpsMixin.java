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

import org.flag4j.core_temp.arrays.dense.DenseVectorMixin;

/**
 * This interface specifies operations between a matrix and a vector that a matrix should implement.
 * @param <T> Type of the matrix.
 * @param <U> Type of the vector.
 * @param <V> Type of dense vector equivalent to {@code U}. If {@code U} is dense, than {@code V} should be the same type as {@code U}.
 */
public interface MatrixVectorOpsMixin<T extends MatrixMixin, U extends VectorMixin, V extends DenseVectorMixin> {

    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the
     * number of entries in the vector {@code b}.
     */
    public V mult(U b);


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     * @return A vector equivalent to this matrix.
     */
    public U toVector();


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     * @return The specified row of this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     * the number of rows in this matrix.
     */
    public U getRow(int rowIdx);


    /**
     * Gets a specified row of this matrix between {@code colStart} (inclusive) and {@code colEnd} (exclusive).
     * @param rowIdx Index of the row of this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     * @return The row at index {@code rowIdx} of this matrix between the {@code colStart} and {@code colEnd}
     * indices.
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException If {@code colEnd} is less than {@code colStart}.
     */
    public U getRow(int rowIdx, int colStart, int colEnd);


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     * @return The specified column of this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     * the number of columns in this matrix.
     */
    public U getCol(int colIdx);


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     * @param colIdx Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     * @throws @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the
     * shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    public U getCol(int colIdx, int rowStart, int rowEnd);


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     * @return A vector containing the diagonal entries of this matrix.
     */
    public U getDiag();


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    public T setCol(U values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    public T setRow(U values, int rowIndex);
}
