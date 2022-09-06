package com.flag4j;


import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations which should be implemented by any matrix or vector.
 * @param <T> Tensor type.
 * @param <U> Dense Tensor type.
 * @param <V> Sparse Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type
 * @param <X> Tensor element type.
 */
interface Operations<T, U, V, W, Y, X> {


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
    public W scalMult(CNumber factor);


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
    public W scalDiv(CNumber divisor);


    /**
     * Sums together all entries in the tensor.
     * @return The sum of all entries in this matrix.
     */
    public X sum();


    /**
     * Computes the element-wise square root of a tensor.
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    public W sqrt();


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    public Y abs();


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     * @return The transpose of this tensor.
     */
    public T transpose();


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * @return The transpose of this tensor.
     */
    public T T();


    /**
     * Computes the complex conjugate of a tensor.
     * @return The complex conjugate of this tensor.
     */
    public T conj();


    /**
     * Computes the complex conjugate transpose of a tensor.
     * Same as {@link #hermTranspose()} and {@link #hermTranspose()}.
     * @return The complex conjugate transpose of this tensor.
     */
    public T conjT();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link #H()}.
     * @returnT he complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    public T hermTranspose();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link #hermTranspose()}.
     * @returnT he complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    public T H();
}
