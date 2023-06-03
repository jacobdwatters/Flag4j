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

import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies methods which any complex tensor should implement.
 * @param <T> Tensor type.
 * @param <Y> Real tensor type.
 */
public interface ComplexTensorMixin<T, Y> {

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
     * Computes the complex conjugate of a tensor.
     * @return The complex conjugate of this tensor.
     */
    T conj();


    /**
     * Converts a complex tensor to a real tensor. The imaginary component of any complex value will be ignored.
     * @return A tensor of the same size containing only the real components of this tensor.
     */
    Y toReal();


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     * @return The complex transpose of this tensor.
     */
    T hermTranspose();


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     * @return The complex transpose of this tensor.
     */
    T H();


    /**
     * Sets an index of this tensor to a specified value.
     * @param value Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException If the number of indices is not equal to the rank of this tensor.
     * @throws IndexOutOfBoundsException If any of the indices are not within this tensor.
     */
    T set(CNumber value, int... indices);
}
