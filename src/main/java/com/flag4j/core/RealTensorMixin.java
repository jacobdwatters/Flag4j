package com.flag4j.core;


/**
 * This interface specifies methods which all real tensors should implement.
 * @param <T> Matrix type.
 * @param <W> Complex matrix type.
 */
public interface RealTensorMixin<T, W> {

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
