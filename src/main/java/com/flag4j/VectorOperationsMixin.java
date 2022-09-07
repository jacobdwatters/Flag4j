package com.flag4j;


import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations which should be implemented by any vector.
 * @param <T> Vector type.
 * @param <U> Dense Vector type.
 * @param <V> Sparse Vector type.
 * @param <W> Complex Vector type.
 * @param <Y> Real Vector type.
 * @param <X> Vector entry type.
 * @param <TT> Matrix type equivalent.
 * @param <UU> Dense Matrix type equivalent.
 */
public interface VectorOperationsMixin<T, U, V, W, Y, X, TT, UU> {


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    public X innerProduct(Vector b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    public X innerProduct(SparseVector b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    public CNumber innerProduct(CVector b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    public CNumber innerProduct(SparseCVector b);


    /**
     * Computes the vector cross product between two vectors.
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public U cross(Vector b);


    /**
     * Computes the vector cross product between two vectors.
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public U cross(SparseVector b);


    /**
     * Computes the vector cross product between two vectors.
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVector cross(CVector b);


    /**
     * Computes the vector cross product between two vectors.
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVector cross(SparseCVector b);


    /**
     * Converts a vector to an equivalent matrix.
     * @return
     */
    public UU toMatrix();
}
