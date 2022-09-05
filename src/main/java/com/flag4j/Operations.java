package com.flag4j;


import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations which should be implemented by any matrix or vector.
 * @param <T> Tensor type.
 * @param <U> Dense Tensor type.
 * @param <V> Complex Tensor type.
 * @param <X> Tensor element type.
 */
interface Operations<T, U, V, X> {


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public T add(T B);


    /**
     * Adds specified value to all entries of this tensor.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    public U add(double a);


    /**
     * Adds specified value to all entries of this tensor.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    public CMatrix add(CNumber a);


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public T sub(T B);


    /**
     * Adds specified value to all entries of this tensor.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    public U sub(double a);


    /**
     * Subtracts a specified value from all entries of this tensor.
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    public CMatrix sub(CNumber a);


    /**
     * Computes scalar multiplication of a tensor.
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    public T scalMult(double factor);


    /**
     * Computes scalar multiplication of a tensor.
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    public V scalMult(CNumber factor);


    /**
     * Computes the scalar division of a tensor.
     * @param divisor The scaler value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    public T scalDiv(double divisor);


    /**
     * Computes the scalar division of a tensor.
     * @param divisor The scaler value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    public V scalDiv(CNumber divisor);


    /**
     * Sums together all entries in the tensor.
     * @return The sum of all entries in this matrix.
     */
    public X sum();
}
