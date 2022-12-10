package com.flag4j.core;


/**
 * This interface specifies manipulations which all tensors (i.e. matrices and vectors) should implement.
 *
 * @param <T> Tensor type.
 * @param <U> Dense Tensor type.
 * @param <V> Sparse Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type.
 * @param <X> Tensor entry type.
 */
interface TensorManipulationsMixin<T, U, V, W, Y, X extends Number> {

    /**
     * Sets an index of this tensor to a specified value.
     * @param value Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    void set(double value, int... indices);
}
