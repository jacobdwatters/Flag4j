package com.flag4j.concurrency;


import com.flag4j.TypedMatrix;
import com.flag4j.util.ErrorMessages;


/**
 * A class for applying concurrent operations to matrices.
 * @param <T> Type of the entries of the matrix.
 * @param <U> Type of the matrix.
 */
public final class ConcurrentOperation<T extends Number, U extends TypedMatrix<T>> {

    /**
     * Private constructor to hide from other files.
     */
    private ConcurrentOperation() {
        // Hides constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Applies a closed element-wise binary operation to two matrices concurrently.
     * @param operation Operation to apply element-wise.
     * @param A First matrix in binary operation.
     * @param B Second matrix in binary operation.
     * @return The result of the element-wise binary operation
     */
    public U applyElementWiseConcurrent(BinaryOperation<T> operation, U A, U B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Applies a closed element-wise unary operation to a matrix concurrently.
     * @param operation Operation to apply element-wise.
     * @param A Matrix in operation.
     * @return The result of the element-wise unary operation.
     */
    public U applyElementWiseConcurrent(UnaryOperation<T> operation, U A) {
        // TODO: Implementation
        return null;
    }
}
