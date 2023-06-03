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

package com.flag4j.core.sparse;

import com.flag4j.Shape;
import com.flag4j.core.RealTensorMixin;
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.operations.dense.real.RealDenseProperties;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;
import com.flag4j.util.SparseDataWrapper;

import java.math.BigInteger;

/**
 * Base class for all sparse tensor.
 * @param <T> Type of this tensor.
 * @param <U> Dense Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Z> Dense complex tensor type.
 */
public abstract class RealSparseTensorBase<T, U, W, Z>
        extends SparseTensorBase<T, U, W, Z, T, double[], Double>
        implements RealTensorMixin<T, W> {

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
    public RealSparseTensorBase(Shape shape, int nonZeroEntries, double[] entries, int[][] indices) {
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
        return AggregateReal.min(entries);
    }


     @Override
     public double max() {
        return AggregateReal.max(entries);
    }


     @Override
     public double minAbs() {
        return AggregateReal.minAbs(entries);
    }


     @Override
     public double maxAbs() {
        return AggregateReal.maxAbs(entries);
    }


     @Override
     public double infNorm() {
        return AggregateReal.maxAbs(entries);
    }


     @Override
     public boolean isPos() {
        return RealProperties.isPos(entries);
    }


     @Override
     public boolean isNeg() {
        return RealProperties.isNeg(entries);
    }


     @Override
     public boolean isZeros() {
         return entries.length==0 || ArrayUtils.isZeros(entries);
    }


    /**
     * Checks if this sparse tensors non-zero values are all ones.
     * @return True if this sparse tensors non-zero values are all ones. False otherwise.
     */
    @Override
    public boolean isOnes() {
        return RealDenseProperties.isOnes(entries);
    }
}
