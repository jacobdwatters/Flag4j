package com.flag4j.concurrency;

/**
 * An interface for creating operations which require no parameters.
 */
public interface ZeroOperation<T> {

    /**
     * Applies a zero parameter operation.
     * @return The result of the zero parameter operation.
     */
    T apply();
}
