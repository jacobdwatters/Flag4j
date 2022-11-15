package com.flag4j;

/**
 * This interface specifies methods which provide properties of a tensor. All tensors should implement this interface.
 *
 * @param <T> Tensor type.
 * @param <U> Dense Tensor type.
 * @param <V> Sparse Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type.
 * @param <X> Tensor entry type.
 */
interface TensorPropertiesMixin<T, U, V, W, Y, X extends Number> {


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    X min();


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    X max();


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     * @return The minimum value, in absolute value, in this tensor.
     */
    X minAbs();


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     * @return The maximum value, in absolute value, in this tensor.
     */
    X maxAbs();


    /**
     * Finds the indices of the minimum value in this tensor.
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argMin();


    /**
     * Finds the indices of the maximum value in this tensor.
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argMax();


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     * @return the 2-norm of this tensor.
     */
    X norm();


    /**
     * Computes the p-norm of this tensor.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    X norm(double p);


    /**
     * Computes the maximum/infinite norm of this tensor.
     * @return The maximum/infinite norm of this tensor.
     */
    X infNorm();
}
