package com.flag4j;


import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations which should be implemented by any matrix (rank 2 tensor).
 * @param <T> Vector type.
 * @param <U> Dense Vector type.
 * @param <V> Complex Vector type.
 * @param <W> Vector entry type.
 */
public interface VectorOperations<T, U, V, W> extends Operations<T, U, V> {


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    public W innerProduct(Vector b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    public W innerProduct(SparseVector b);


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
     * Computes the element-wise multiplication (Hadamard product) between two Vectors.
     * @param b Second vector in the element-wise multiplication.
     * @return The result of element-wise multiplication of this vector with the vector b.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public T elemMult(Vector b);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two Vectors.
     * @param b Second vector in the element-wise multiplication.
     * @return The result of element-wise multiplication of this vector with the vector b.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public V elemMult(SparseVector b);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two Vectors.
     * @param b Second vector in the element-wise multiplication.
     * @return The result of element-wise multiplication of this vector with the vector b.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public W elemMult(CVector b);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two Vectors.
     * @param b Second vector in the element-wise multiplication.
     * @return The result of element-wise multiplication of this vector with the vector b.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public SparseCVector elemMult(SparseCVector b);


    /**
     * Computes the element-wise division between two vectors.
     * @param b Second vector in the element-wise division.
     * @return The result of element-wise division of this vector with the vector b.
     * @throws IllegalArgumentException If this vector and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    public T elemDiv(Vector b);


    /**
     * Computes the element-wise division between two vectors.
     * @param b Second vector in the element-wise division.
     * @return The result of element-wise division of this vector with the vector b.
     * @throws IllegalArgumentException If this vector and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    public V elemDiv(CVector b);
}
