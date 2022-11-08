package com.flag4j.concurrency;


/**
 * Interface for creating a closed binary operation lambda for use in the {@link ConcurrentOperation} class.
 * @param <T> Type to perform the operation on. Must extend {@link Number}.
 */
public interface BinaryOperation<T extends Number> {


    /**
     * Applies a closed binary operation on the two inputs.
     * @param a First input to binary operation.
     * @param b Second input to the binary operation.
     * @return The result of applying a binary operation
     */
    T apply(T a, T b);
}
