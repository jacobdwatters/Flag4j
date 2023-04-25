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

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.operations.dense.real.RealDenseSetOperations;
import com.flag4j.util.ParameterChecks;


/**
 * The base class for all real matrices.
 * @param <T> Type of this matrix.
 * @param <W> Type of complex type equivalent of this matrix.
 */
public abstract class RealMatrixBase<T, W>
        extends MatrixBase<T, Matrix, CMatrix, T, double[], Double>
        implements RealMatrixMixin<T, W> {


    /**
     * Constructs a basic matrix with a given shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     * @throws IllegalArgumentException If the shape parameter is not of rank 2.
     */
    public RealMatrixBase(Shape shape, double[] entries) {
        super(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent complex matrix.
     * @return A complex matrix with equivalent real part and zero imaginary part.
     */
    public abstract W toComplex();


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    public RealMatrixBase<T, W> setValues(Integer[][] values) {
        ParameterChecks.assertEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
        return this;
    }
}
