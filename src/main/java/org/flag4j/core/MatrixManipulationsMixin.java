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

package org.flag4j.core;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;

/**
 * This interface specifies manipulations which all matrices should implement.
 *
 * @param <T> Matrix type.
 * @param <X> Matrix entry type.
 */
public interface MatrixManipulationsMixin<
        T,
        X extends Number> {

    /**
     * Sets an index of this matrix to the specified value.
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     * @return A reference to this matrix.
     */
    T set(double value, int row, int col);


    /**
     * Sets an index of this matrix to the specified value.
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     * @return A reference to this matrix.
     */
    T set(X value, int row, int col);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    T setCol(X[] values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    T setCol(Double[] values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    T setCol(Integer[] values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    T setCol(double[] values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    T setCol(int[] values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    T setRow(X[] values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    T setRow(Double[] values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    T setRow(Integer[] values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    T setRow(double[] values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    T setRow(int[] values, int rowIndex);

    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(T values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(Matrix values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(CooMatrix values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(X[][] values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(Double[][] values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(Integer[][] values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(double[][] values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(int[][] values, int rowStart, int colStart);


    /**
     * Removes a specified row from this matrix.
     * @param rowIndex Index of the row to remove from this matrix.
     * @return a copy of this matrix with the specified row removed.
     */
    T removeRow(int rowIndex);


    /**
     * Removes a specified set of rows from this matrix.
     * @param rowIndices The indices of the rows to remove from this matrix.
     * @return a copy of this matrix with the specified rows removed.
     */
    T removeRows(int... rowIndices);


    /**
     * Removes a specified column from this matrix.
     * @param colIndex Index of the column to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    T removeCol(int colIndex);


    /**
     * Removes a specified set of columns from this matrix.
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return a copy of this matrix with the specified columns removed.
     */
    T removeCols(int... colIndices);


    /**
     * Swaps rows in the matrix.
     * @param rowIndex1 Index of first row to swap.
     * @param rowIndex2 index of second row to swap.
     * @return A reference to this matrix.
     */
    T swapRows(int rowIndex1, int rowIndex2);


    /**
     * Swaps columns in the matrix.
     * @param colIndex1 Index of first column to swap.
     * @param colIndex2 index of second column to swap.
     * @return A reference to this matrix.
     */
    T swapCols(int colIndex1, int colIndex2);
}
