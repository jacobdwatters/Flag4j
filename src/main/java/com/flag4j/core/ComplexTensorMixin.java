/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j.core;

import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.SparseCVector;
import com.flag4j.SparseMatrix;
import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies methods which any complex tensor should implement.
 * @param <T> Matrix type.
 * @param <Y> Real matrix type.
 */
interface ComplexTensorMixin<T, Y> {

    /**
     * Checks if this tensor has only real valued entries.
     * @return True if this tensor contains <b>NO</b> complex entries. Otherwise, returns false.
     */
    boolean isReal();


    /**
     * Checks if this tensor contains at least one complex entry.
     * @return True if this tensor contains at least one complex entry. Otherwise, returns false.
     */
    boolean isComplex();


    /**
     * Converts a complex tensor to a real matrix. The imaginary component of any complex value will be ignored.
     * @return A tensor of the same size containing only the real components of this tensor.
     */
    Y toReal();


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    void setValues(CNumber[][] values);


    /**
     * Sets an index of this tensor to a specified value.
     * @param value Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    void set(CNumber value, int... indices);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    void setCol(CNumber[] values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    void setRow(CNumber[] values, int rowIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     */
    void setCol(CVector values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the column.
     * @param colIndex The index of the columns which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     */
    void setCol(SparseCVector values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of columns of this matrix.
     */
    void setRows(CVector values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of columns of this matrix.
     */
    void setRows(SparseCVector values, int rowIndex);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    void setSlice(Matrix values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    void setSlice(SparseMatrix values, int rowStart, int colStart);
}
