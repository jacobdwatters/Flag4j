package com.flag4j.core;

import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;

/**
 * Base class for all sparse tensor.
 * @param <T> Type of the entries of the sparse tensor.
 */
public abstract class SparseTensorBase<T> extends TensorBase<T> {

    /**
     * Indices for non-zero entries of this tensor. Will have shape (rank-by-nonZeroEntries)
     */
    public final int[][] indices;
    private final int nonZeroEntries;


    /**
     * Creates a sparse tensor with specified shape.
     * @param shape Shape of this tensor.
     * @param nonZeroEntries Number of non-zero entries in the sparse tensor.
     */
    public SparseTensorBase(Shape shape, int nonZeroEntries, T entries, int[][] indices) {
        super(shape, entries);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ShapeChecks.arrayLengthsCheck(nonZeroEntries, indices.length);
        if (indices.length > 0) {
            ShapeChecks.arrayLengthsCheck(super.getRank(), indices[0].length);
        }

        this.nonZeroEntries = nonZeroEntries;
        this.indices = indices;
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
