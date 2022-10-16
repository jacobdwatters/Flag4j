package com.flag4j;


/**
 * This interface specifies manipulations which all vectors should implement.
 *
 * @param <T> Vectors type.
 * @param <U> Dense vectors type.
 * @param <V> Sparse vectors type.
 * @param <W> Complex vectors type.
 * @param <Y> Real vectors type.
 * @param <X> Vectors entry type.
 * @param <TT> Matrix type equivalent.
 * @param <UU> Dense Matrix type equivalent.
 * @param <VV> Sparse Matrix type equivalent.
 * @param <WW> Complex Matrix type equivalent.
 */
interface VectorManipulationsMixin<T, U, V, W, Y, X extends Number, TT, UU, VV, WW, YY> extends
        TensorManipulationsMixin<T, U, V, W, Y, X> {


    /**
     * Extends a vector a specified number of times to a matrix.
     * @param n The number of times to extend this vector.
     * @return A matrix which is the result of extending a vector 'n' times.
     */
    TT extend(int n);
}
