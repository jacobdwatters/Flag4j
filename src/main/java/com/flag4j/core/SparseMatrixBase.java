package com.flag4j.core;

import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;

/**
 * Base class for all sparse matrices.
 */
public abstract class SparseMatrixBase<T> extends MatrixBase<T> {

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
    public SparseMatrixBase(Shape shape, int nonZeroEntries, T entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ShapeChecks.arrayLengthsCheck(nonZeroEntries, rowIndices.length, colIndices.length);

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