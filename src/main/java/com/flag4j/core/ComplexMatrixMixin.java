/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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
import com.flag4j.SparseCMatrix;
import com.flag4j.SparseCVector;
import com.flag4j.complex_numbers.CNumber;


/**
 * This interface specifies methods which any complex matrix should implement.
 * @param <T> Matrix type.
 */
public interface ComplexMatrixMixin<T> {


    /**
     * Computes the complex conjugate transpose of a tensor.
     * Same as {@link #hermTranspose()} and {@link MatrixOperationsMixin#H()}.
     * @return The complex conjugate transpose of this tensor.
     */
    T conjT();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link MatrixOperationsMixin#H()}.
     * @return The complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    T hermTranspose();


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is equal to its conjugate transpose.
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    boolean isHermitian();


    /**
     * Checks if a matrix is anti-Hermitian. That is, if the matrix is equal to the negative of its conjugate transpose.
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    boolean isAntiHermitian();


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is equal to its conjugate transpose.
     * @return True if this matrix it is unitary. Otherwise, returns false.
     */
    boolean isUnitary();


    /**
     * Checks if this tensor has only real valued entries.
     *
     * @return True if this tensor contains <b>NO</b> complex entries. Otherwise, returns false.
     */
    boolean isReal();


    /**
     * Checks if this tensor contains at least one complex entry.
     *
     * @return True if this tensor contains at least one complex entry. Otherwise, returns false.
     */
    boolean isComplex();


    /**
     * Computes the complex conjugate of a tensor.
     *
     * @return The complex conjugate of this tensor.
     */
    T conj();


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    T setValues(CNumber[][] values);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex} is not within the matrix.
     */
    T setCol(CNumber[] values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    T setRow(CNumber[] values, int rowIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex} is not within the matrix.
     */
    T setCol(CVector values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the column.
     * @param colIndex The index of the columns which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex} is not within the matrix.
     */
    T setCol(SparseCVector values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    T setRow(CVector values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    T setRow(SparseCVector values, int rowIndex);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSlice(SparseCMatrix values, int rowStart, int colStart);


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified {@code values}.
     * @throws IndexOutOfBoundsException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException  If the {@code values} slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    T setSliceCopy(SparseCMatrix values, int rowStart, int colStart);


    /**
     * Adds a complex sparse matrix to this matrix and stores the result in this matrix.
     *
     * @param B Complex sparse matrix to add to this matrix,
     */
    void addEq(SparseCMatrix B);


    /**
     * Subtracts a complex sparse matrix from this matrix and stores the result in this matrix.
     *
     * @param B Complex sparse matrix to subtract from this matrix,
     */
    void subEq(SparseCMatrix B);
}
