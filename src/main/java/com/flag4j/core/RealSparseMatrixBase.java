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
import com.flag4j.SparseMatrix;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;

/**
 * Base class for all real sparse matrices.
 */
public abstract class RealSparseMatrixBase extends RealMatrixBase<SparseMatrix, SparseCMatrix> {

    /**
     * Row indices.
     */
    public final int[] rowIndices;
    /**
     * Col indices.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero entries in this sparse matrix.
     */
    private int nonZeroEntries;


    /**
     * Creates a sparse matrix with specified size, non-zero entries, and row/column indices.
     * @param shape Shape of this sparse matrix.
     * @param nonZeroEntries Number of non-zero entries in the sparse matrix.
     * @param entries Non-zero entries of this sparse tensor.
     * @param rowIndices The row indices of all non-zero entries.
     * @param colIndices The column indices of all non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries, row indices, and column indices are not all
     * equal.
     */
    public RealSparseMatrixBase(Shape shape, int nonZeroEntries, double[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ParameterChecks.assertArrayLengthsEq(nonZeroEntries, rowIndices.length, colIndices.length);

        this.nonZeroEntries = nonZeroEntries;
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
    }


    /**
     * Sets the number of non-zero entries in this sparse matrix.
     * @param nonZeroEntries New number of non-zero entries in this sparse matrix.
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
     * Gets the sparsity of this matrix as a decimal percentage.
     * @return The sparsity of this matrix.
     */
    public double sparsity() {
        BigDecimal sparsity = new BigDecimal(this.totalEntries()).subtract(BigDecimal.valueOf(this.nonZeroEntries()));
        sparsity = sparsity.divide(new BigDecimal(this.totalEntries()), 50,RoundingMode.HALF_UP);

        return sparsity.doubleValue();
    }


    /**
     * Gets the density of this matrix as a decimal percentage.
     * @return The density of this matrix.
     */
    public double density() {
        BigDecimal density = BigDecimal.valueOf(this.nonZeroEntries).divide(
                new BigDecimal(this.totalEntries()), 50, RoundingMode.HALF_UP);

        return density.doubleValue();
    }

    // TODO: Add abstract methods for sparse matrices. i.e. toDense().
}
