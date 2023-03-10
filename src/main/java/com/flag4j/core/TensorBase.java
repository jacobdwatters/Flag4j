/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
     * Entry data for this tensor.
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
     * <p>
     * Gets the rank of this tensor. That is, number of indices needed to uniquely select an element of the tensor.
     * </p>
     *
     * <p>
     * Note, this method is distinct from the {@link MatrixPropertiesMixin#matrixRank()} method.
     * This returns the number of dimensions (i.e. order or degree) of the tensor and indicates the number of indices
     * needed to uniquely select an element of the tensor.
     * </p>
     *
     * @return The rank of this tensor.
     */
    public int getRank() {
        return this.shape.getRank();
    }


    /**
     * Gets the entries of this tensor as a 1D array.
     * @return The entries of this tensor.
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
    public boolean sameShape(TensorBase<?> B) {
        return this.shape.equals(B.shape);
    }


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param A First tensor to compare.
     * @param B Second tensor to compare.
     * @param axis The axis along which to compare the lengths of the two tensors.
     * @return True if tensor A and tensor B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is negative or unspecified for either tensor.
     */
    public static boolean sameLength(TensorBase<?> A, TensorBase<?> B, int axis) {
        if(axis < 0 || axis>=Math.min(A.shape.getRank(), B.shape.getRank())) {
            throw new IllegalArgumentException(
                    ErrorMessages.axisErr(axis)
            );
        }

        return A.shape.dims[axis]==B.shape.dims[axis];
    }
}
