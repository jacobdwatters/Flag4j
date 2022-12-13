package com.flag4j.core;


import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;

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
    private int nonZeroEntries;


    /**
     * Creates a sparse vector with specified number of entries.
     * @param totalEntries Number of total entries in this sparse vector, including zeros.
     * @param nonZeroEntries Number of non-zero entries in this sparse vector.
     * @param orientation Orientation of this sparse vector.
     */
    public SparseVectorBase(int totalEntries, int nonZeroEntries, VectorOrientations orientation, T entries, int[] indices) {
        super(totalEntries, orientation, entries);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ShapeChecks.arrayLengthsCheck(nonZeroEntries, indices.length);

        this.nonZeroEntries = nonZeroEntries;
        this.indices = indices;
    }


    /**
     * Sets the number of nonZero entries in this sparse vector.
     * @param nonZeroEntries
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