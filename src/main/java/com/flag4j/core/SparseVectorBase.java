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


import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;

/**
 * Base class for all sparse vectors.
 * @param <T> Type of the entries for this sparse vector.
 */
public abstract class SparseVectorBase<T> extends VectorBase<T> {

    /**
     * Indices of non-zero values in this sparse vector.
     */
    public final int[] indices;
    /**
     * Number of non-zero entries of this sparse vec
     */
    private int nonZeroEntries;


    /**
     * Creates a sparse vector with specified number of entries.
     * @param totalEntries Number of total entries in this sparse vector, including zeros.
     * @param nonZeroEntries Number of non-zero entries in this sparse vector.
     * @param orientation Orientation of this sparse vector.
     * @param entries Non-zero entries of this sparse vector.
     * @param indices Indices of the non-zero entries of this tensor.
     * @throws IllegalArgumentException If the lengths of the entries and incicies arrays are not equal.
     */
    public SparseVectorBase(int totalEntries, int nonZeroEntries, VectorOrientation orientation, T entries, int[] indices) {
        super(totalEntries, orientation, entries);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ParameterChecks.assertArrayLengthsEq(nonZeroEntries, indices.length);

        this.nonZeroEntries = nonZeroEntries;
        this.indices = indices;
    }


    /**
     * Sets the number of non-zero entries in this sparse vector. WARNING: Caution should be used when calling this
     * method.
     * @param nonZeroEntries Non-zero entries contained within this sparse vector.
     */
    protected void setNonZeroEntries(int nonZeroEntries) {
        this.nonZeroEntries = nonZeroEntries;
    }


    /**
     * Gets the number of non-zero entries in this sparse matrix.
     * @return The number of non-zero entries in this sparse matrix.
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
