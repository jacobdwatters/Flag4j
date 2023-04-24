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

import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
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
     * Sets an index of this matrix to the specified value.
     * @param value New value.
     * @param indices Indices for new value.
     * @return A reference to this matrix.
     */
    public abstract MatrixBase<CNumber[]> set(CNumber value, int... indices);


    /**
     * Adds a complex sparse matrix to this matrix and stores the result in this matrix.
     * @param B Complex sparse matrix to add to this matrix,
     */
    public abstract void addEq(SparseCMatrix B);


    /**
     * Subtracts a complex sparse matrix from this matrix and stores the result in this matrix.
     * @param B Complex sparse matrix to subtract from this matrix,
     */
    public abstract void subEq(SparseCMatrix B);
}
