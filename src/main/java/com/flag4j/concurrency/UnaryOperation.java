package com.flag4j.concurrency;


/**
 * Interface for creating a closed unary operation lambda for use in the {@link ConcurrentOperation} class.
 * @param <T> Type to perform the operation on. Must extend {@link Number}.
 */
public interface UnaryOperation<T extends Number> {

    /**
     * Applies a closed unary operation to the input parameter.
     * @param a
     * @return
     */
    T apply(T a);
}


