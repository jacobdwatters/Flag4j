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

import com.flag4j.CMatrix;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;


/**
 * This interface specifies methods which any complex matrix should implement.
 * @param <T> Matrix type.
 * @param <Y> Real matrix type.
 */
interface ComplexMatrixMixin<T, Y> extends
        ComplexTensorMixin<T, Y>,
        MatrixPropertiesMixin<T, CMatrix, SparseCMatrix, T, Y, CNumber>,
        MatrixOperationsMixin<T, CMatrix, SparseCMatrix, T, Y, CNumber> {


    /**
     * Computes the complex conjugate of a tensor.
     * @return The complex conjugate of this tensor.
     */
    T conj();


    /**
     * Computes the complex conjugate transpose of a tensor.
     * Same as {@link #hermTranspose()} and {@link #hermTranspose()}.
     * @return The complex conjugate transpose of this tensor.
     */
    T conjT();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link #H()}.
     * @return he complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    T hermTranspose();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link #hermTranspose()}.
     * @return he complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    T H();


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
}
