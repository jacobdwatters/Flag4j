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
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.operations.dense.real.RealDenseSetOperations;
import com.flag4j.util.ParameterChecks;


/**
 * The base class for all real matrices.
 * @param <T> Type of this matrix.
 * @param <W> Type of complex type equivalent of this matrix.
 */
public abstract class RealMatrixBase<T, W>
        extends MatrixBase<T, Matrix, CMatrix, CMatrix, T, double[], Double>
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


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public double min() {
        return AggregateReal.min(entries);
    }


    /**
     * Finds the maximum value in this matrix. If this matrix has zero entries, the method will return 0.
     * @return The maximum value in this matrix.
     */
    @Override
    public double max() {
        return AggregateReal.max(entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return AggregateReal.minAbs(entries);
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return AggregateReal.maxAbs(entries);
    }


    /**
     * Checks if this tensor contains only non-negative values.
     *
     * @return True if this tensor only contains non-negative values. Otherwise, returns false.
     */
    @Override
    public boolean isPos() {
        return RealProperties.isPos(entries);
    }


    /**
     * Checks if this tensor contains only non-positive values.
     *
     * @return trie if this tensor only contains non-positive values. Otherwise, returns false.
     */
    @Override
    public boolean isNeg() {
        return RealProperties.isNeg(entries);
    }
}
