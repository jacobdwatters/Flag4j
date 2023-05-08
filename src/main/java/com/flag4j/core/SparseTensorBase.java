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
import com.flag4j.util.ParameterChecks;

import java.math.BigDecimal;
import java.math.RoundingMode;


/**
 * The base class for all sparse tensors. This includes sparse matrices and vectors.
 * @param <T> Type of this tensor.
 * @param <U> Dense Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Z> Dense complex tensor type.
 * @param <Y> Real Tensor type.
 * @param <D> Type of the storage data structure for the tensor.
 *           This common use case will be an array or list-like data structure.
 * @param <X> The type of individual entry within the {@code D} data structure
 */
public abstract class SparseTensorBase<T, U, W, Z, Y, D, X extends Number>
        extends TensorBase<T, U, W, Z, Y, D, X>
        implements SparseTensorMixin {

    /**
     * Indices for non-zero entries of this tensor. Will have shape {@code (nonZeroEntries-by-rank)}.
     */
    public final int[][] indices;
    /**
     * The number of non-zero entries in this sparse tensor.
     */
    protected final int nonZeroEntries;

    /**
     * Creates a sparse tensor with specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape   Shape of this tensor.
     * @param entries Non-zero entries of this tensor.
     * @param indices Indices of non-zero entries in this tensor. Must have shape {@code (nonZeroEntries-by-rank)}.
     * @throws IllegalArgumentException If the rank of {@code shape} does not match the number of columns in {@code indices}.
     */
    public SparseTensorBase(Shape shape, int nonZeroEntries, D entries, int[][] indices) {
        super(shape, entries);

        if(indices.length > 0) {
            ParameterChecks.assertEquals(indices[0].length, shape.getRank());
        }

        this.indices = indices;
        this.nonZeroEntries = nonZeroEntries;
    }


    /**
     * Gets the number of non-zero entries in this sparse tensor.
     * @return The number of non-zero entries in this sparse tensor.
     */
    public int nonZeroEntries() {
        return nonZeroEntries;
    }


    /**
     * Gets the sparsity of this tensor as a decimal percentage.
     * @return The sparsity of this tensor.
     */
    public double sparsity() {
        BigDecimal sparsity = new BigDecimal(this.totalEntries()).subtract(BigDecimal.valueOf(this.nonZeroEntries()));
        sparsity = sparsity.divide(new BigDecimal(this.totalEntries()), 50, RoundingMode.HALF_UP);

        return sparsity.doubleValue();
    }


    /**
     * Gets the density of this tensor as a decimal percentage.
     * @return The density of this tensor.
     */
    public double density() {
        BigDecimal density = BigDecimal.valueOf(this.nonZeroEntries).divide(
                new BigDecimal(this.totalEntries()), 50, RoundingMode.HALF_UP
        );

        return density.doubleValue();
    }
}
