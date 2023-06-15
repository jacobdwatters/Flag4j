package com.flag4j.exceptions;


/**
 * An exception which is thrown when a linear algebra related error occurs at runtime.
 */
public class LinearAlgebraException extends RuntimeException {

    /**
     * Creates a {@link LinearAlgebraException} to be thrown for a linear algebra related error.
     * @param errMsg Error message for the exception.
     */
    public LinearAlgebraException(String errMsg) {
        super(errMsg);
    }
}
