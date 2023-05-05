/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.common.complex.AggregateComplex;
import com.flag4j.operations.dense.complex.ComplexDenseProperties;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ParameterChecks;

/**
 * The base class for all complex dense tensors. This includes complex dense matrices and vectors.
 * @param <T> Type of this tensor.
 * @param <Y> Real Tensor type.
 */
public abstract class ComplexDenseTensorBase<T, Y>
        extends DenseTensorBase<T, T, Y, CNumber[], CNumber>
        implements ComplexTensorMixin<T, Y> {


    /**
     * Creates a complex dense tensor with specified entries and shape.
     *
     * @param shape   Shape of this tensor.
     * @param entries Entries of this tensor. The number of entries must match the product of
     *                all {@code shape} dimensions.
     * @throws IllegalArgumentException If the number of entries does not equal the product of dimensions in the
     * {@code shape}.
     */
    public ComplexDenseTensorBase(Shape shape, CNumber[] entries) {
        super(shape, entries);
        ParameterChecks.assertEquals(shape.totalEntries().intValueExact(), entries.length);
    }

    // TODO: Pull up implementation of all methods which are the same for any real dense tensor.
    //  e.g. addition/subtraction, element-wise/scalar multiplication/division, any element-wise operation
    //  such as abs, sqrt, etc.


    @Override
    public double min() {
        return minAbs();
    }


    @Override
    public double max() {
        return maxAbs();
    }


    @Override
    public double minAbs() {
        return AggregateComplex.minAbs(entries);
    }


    @Override
    public double maxAbs() {
        return AggregateComplex.maxAbs(entries);
    }


    @Override
    public double infNorm() {
        return AggregateComplex.maxAbs(entries);
    }


    @Override
    public boolean isZeros() {
        return ArrayUtils.isZeros(entries);
    }


    @Override
    public boolean isOnes() {
        return ComplexDenseProperties.isOnes(entries);
    }
}
