package com.flag4j.core;


/**
 * This interface specifies comparisons which all tensors (i.e. matrices and vectors) should implement.
 *
 * @param <T> Tensor type.
 * @param <U> Dense Tensor type.
 * @param <V> Sparse Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type.
 * @param <X> Tensor entry type.
 */
public interface TensorComparisonsMixin<T, U, V, W, Y, X extends Number> {


    /**
     * Checks if this tensor only contains zeros.
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    boolean isZeros();


    /**
     * Checks if this tensor only contains ones.
     * @return True if this tensor only contains oens. Otherwise, returns false.
     */
    boolean isOnes();


    /**
     * Checks if this tensor is equal to a specified Object.
     * @param B Object to compare this tensor to.
     */
    @Override
    boolean equals(Object B);
}
