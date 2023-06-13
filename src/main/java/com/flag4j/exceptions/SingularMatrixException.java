package com.flag4j.exceptions;

/**
 * An exception which is thrown when some operation not defined for singular matrices is attempted to be
 * performed on a singular matrix. For example, attempting to invert a singular matrix.
 */
public class SingularMatrixException extends LinearAlgebraException {

    private static final String INFO = "Matrix is singular.";

    /**
     * Creates a SingularMatrixException with the simple error message "Matrix is singular."
     */
    public SingularMatrixException() {
        super(INFO);
    }

    /**
     * Creates a SingularMatrixException with a specified error message. Note, the string " Matrix is singular." will
     * be automatically appended to the error message.
     * @param errMsg Error message to display when this SingularMatrixException is thrown.
     */
    public SingularMatrixException(String errMsg) {
        super(errMsg + " " + INFO);
    }
}
