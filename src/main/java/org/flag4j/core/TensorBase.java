/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.core;

import org.flag4j.operations_old.common.TensorEquals;
import org.flag4j.util.ErrorMessages;

import java.io.Serializable;
import java.math.BigInteger;


/**
 * The base class for all tensors. A tensor is an algebraic object equivalent to a multidimensional array with a single
 * data type.
 * @param <T> Type of this tensor.
 * @param <U> Dense TensorOld type.
 * @param <W> Complex TensorOld type.
 * @param <Z> Dense complex tensor type.
 * @param <Y> Real TensorOld type.
 * @param <D> Type of the storage data structure for the tensor.
 *           This common use case will be an array or list-like data structure.
 * @param <X> The type of individual entry within the {@code D} data structure
 */
public abstract class TensorBase<T, U, W, Z, Y, D extends Serializable, X extends Number> implements Serializable,
        TensorComparisonsMixin,
        TensorPropertiesMixin,
        TensorManipulationsMixin<T>,
        TensorOperationsMixin<T, U, W, Z, Y, X> {

    /**
     * Default value for rounding to zero.
     */
    protected static final double DEFAULT_ROUND_TO_ZERO_THRESHOLD = 1e-12;

    /**
     * Entry data for this tensor.
     */
    public final D entries;
    /**
     * The shape of this tensor
     */
    public final Shape shape;


    /**
     * Creates a tensor with specified entries and shape.
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     *                If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected TensorBase(Shape shape, D entries) {
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
    public D getEntries() {
        return this.entries;
    }


    /**
     * Gets the total number of entries in this tensor.
     * @return The total number of entries in this tensor.
     */
    public BigInteger totalEntries() {
        // Using the shape to compute the total number of entries ensures the correct result for sparse tensors.
        return shape.totalEntries();
    }


    /**
     * Checks if a tensor has the same shape as this tensor.
     * @param B Second tensor.
     * @return True if this tensor and B have the same shape. False otherwise.
     */
    public boolean sameShape(TensorBase<?, ?, ?, ?, ?, ?, ?> B) {
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
    public static boolean sameLength(TensorBase<?, ?, ?, ?, ?, ?, ?> A, TensorBase<?, ?, ?, ?, ?, ?, ?> B, int axis) {
        if(axis < 0 || axis>=Math.min(A.shape.getRank(), B.shape.getRank())) {
            throw new IllegalArgumentException(
                    ErrorMessages.getAxisErr(axis)
            );
        }

        return A.shape.get(axis)==B.shape.get(axis);
    }


    /**
     * Simply returns a reference of this tensor.
     * @return A reference to this tensor.
     */
    protected abstract T getSelf();


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     * @param tensor TensorOld to compare this tensor to.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (1E-05 + 1E-08*|b|)}. Otherwise, returns false.
     * @see #allClose(Object, double, double)
     */
    public boolean allClose(T tensor) {
        return allClose(tensor, 1e-05, 1e-08);
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     * @param tensor TensorOld to compare this tensor to.
     * @param absTol Absolute tolerance.
     * @param relTol Relative tolerance.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (atol + rtol*|b|)}. Otherwise, returns false.
     * @see #allClose(Object)
     */
    public abstract boolean allClose(T tensor, double relTol, double absTol);


    @Override
    public boolean tensorEquals(TensorBase<?, ?, ?, ?, ?, ?, ?> src2) {
        return TensorEquals.generalEquals(this, src2);
    }
}
