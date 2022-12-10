package com.flag4j.core;

import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;

import java.io.Serializable;
import java.math.BigInteger;


/**
 * A tensor with typed entries.
 * @param <T> Type of the entries of this tensor.
 */
public abstract class TensorBase<T> implements Serializable {

    /**
     * Entries of this tensor.
     */
    public final T entries;
    /**
     * The shape of this tensor
     */
    public final Shape shape;


    /**
     * Creates an empty tensor with given shape.
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public TensorBase(Shape shape, T entries) {
        this.shape = shape;
        this.entries = entries;
    }


    /**
     * Gets the shape of this tensor.
     * @return The shape of this tensor.
     */
    public Shape getShape() {
        return this.shape;
    }


    /**
     * Gets the rank of this tensor.
     * @return The rank of this tensor.
     */
    public int getRank() {
        return this.shape.getRank();
    }


    /**
     * Gets the entries of this tensor as a 1D array.
     * @return
     */
    public T getEntries() {
        return this.entries;
    }


    /**
     * Gets the total number of entries in this tensor.
     * @return The total number of entries in this tensor.
     */
    public BigInteger totalEntries() {
        // Use the shape to compute the number of entries. This ensures the result is also correct for sparse tensors.
        return shape.totalEntries();
    }


    /**
     * Checks if a tensor has the same shape as this tensor.
     * @param B Second tensor.
     * @return True if this tensor and B have the same shape. False otherwise.
     */
    public boolean sameShape(TensorBase B) {
        return this.shape.equals(B.shape);
    }


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param A First tensor to compare.
     * @param B Second tensor to compare.
     * @param axis The axis along which to compare the lengths of the two tensors.
     * @return True if tensor A and tensor B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is negative or unspecified for the two tensors.
     */
    public static boolean sameLength(TensorBase A, TensorBase B, int axis) {
        if(axis < 0 || axis>=Math.max(A.shape.getRank(), B.shape.getRank())) {
            throw new IllegalArgumentException(
                    ErrorMessages.axisErr(axis)
            );
        }

        return A.shape.dims[axis]==B.shape.dims[axis];
    }


    /**
     * Checks if this tensor is empty. That is, has zero size.
     * @return True if this tensor has zero size,
     */
    public boolean isEmpty() {
        boolean result = false;

        for(int dim : shape.dims) {
            if(dim==0) {
                result=true;
                break;
            }
        }

        return result;
    }
}
