package com.flag4j.core;

/**
 * This interface specifies methods which provide properties of a vector. All vectors should implement this interface.
 *
 * @param <T> Vector type.
 * @param <U> Dense Vector type.
 * @param <V> Sparse Vector type.
 * @param <W> Complex Vector type.
 * @param <Y> Real Vector type.
 * @param <X> Vector entry type.
 */
interface VectorPropertiesMixin<T, U, V, W, Y, X extends Number> extends TensorPropertiesMixin<T, U, V, W, Y, X> {

    /**
     * Gets the length of a vector.
     * @return The length, i.e. the number of entries, in this vector.
     */
    int length();
}
