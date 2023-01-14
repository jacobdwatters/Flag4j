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
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;

/**
 * The base class for all complex matrices.
 */
public abstract class ComplexMatrixBase extends MatrixBase<CNumber[]> {


    /**
     * Constructs a basic matrix with a given shape.
     *
     * @param shape   Shape of this matrix.
     * @param entries Entries of this matrix.
     * @throws IllegalArgumentException If the shape parameter is not of rank 2.
     */
    public ComplexMatrixBase(Shape shape, CNumber[] entries) {
        super(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent real matrix. Imaginary components are ignored.
     * @return A real matrix with equivalent real parts.
     */
    public abstract RealMatrixBase toReal();


    /**
     * Computes the hermation transpose (i.e. the conjugate transpose) of the matrix.
     * Same as {@link #H()}.
     * @return The conjugate transpose.
     */
    public abstract CMatrix hermationTranspose();


    /**
     * Computes the hermation transpose (i.e. the conjugate transpose) of the matrix.
     * Same as {@link #hermationTranspose()}.
     * @return The conjugate transpose.
     */
    public abstract CMatrix H();
}
