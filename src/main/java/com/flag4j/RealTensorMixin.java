package com.flag4j;


/**
 * This interface specifies methods which all real tensors should implement.
 * @param <T> Matrix type.
 * @param <W> Complex matrix type.
 */
interface RealTensorMixin<T, W> {

    /**
     * Computes the real square roots of this tensor (element-wise). This method only supports tensors which do not contain
     * negative values. See {@link TensorOperationsMixin#sqrt()} for a method which computes the complex square roots of
     * a tensor.
     * @return A tensor of the same shape as this tensor which is the result of applying a square root (element-wise) to
     * this tensor.
     * @throws IllegalArgumentException If this tensor contains negative values.
     */
    T reSqrt();


    /**
     * Checks if this tensor contains only non-negative values.
     * @return True if this tensor only contains non-negative values. Otherwise, returns false.
     */
    boolean isPos();


    /**
     * Checks if this tensor contains only non-positive values.
     * @return trie if this tensor only contains non-positive values. Otherwise, returns false.
     */
    boolean isNeg();


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link com.flag4j.complex_numbers.CNumber CNumber} rather than {@link Double}.
     * @return A complex matrix which is equivalent to this matrix.
     */
    W toComplex();
}
