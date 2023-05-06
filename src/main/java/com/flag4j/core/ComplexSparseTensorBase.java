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
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;
import com.flag4j.util.SparseDataWrapper;

import java.math.BigInteger;

/**
 * This abstract class is the base class of all complex sparse tensors.
 * @param <T> Type of this tensor.
 * @param <U> Dense Tensor type.
 * @param <Y> Real tensor type.
 */
public abstract class ComplexSparseTensorBase<T, U, Y>
        extends SparseTensorBase<T, U, T, U, Y, CNumber[], CNumber>
        implements ComplexTensorMixin<T, Y> {

    /**
     * Creates a sparse tensor with specified shape.
     * @param shape Shape of this tensor.
     * @param nonZeroEntries Number of non-zero entries in the sparse tensor.
     * @param entries Non-zero entries of this sparse tensor.
     * @param indices Indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of rows in the indices array is not equal to the number of
     * elements in the entries array.
     * @throws IllegalArgumentException If the number of columns in the entries array is not equal to the rank of this
     * tensor.
     */
    public ComplexSparseTensorBase(Shape shape, int nonZeroEntries, CNumber[] entries, int[][] indices) {
        super(shape, nonZeroEntries, entries, indices);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ParameterChecks.assertArrayLengthsEq(nonZeroEntries, indices.length);
    }



    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sparseSort() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Gets the number of non-zero entries stored in this sparse tensor.
     * @return The number of non-zero entries stored in this tensor.
     */
    @Override
    public int nonZeroEntries() {
        return entries.length;
    }


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
