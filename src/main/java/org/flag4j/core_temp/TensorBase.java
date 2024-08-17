/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.core_temp;


import org.flag4j.core.MatrixPropertiesMixin;
import org.flag4j.core.Shape;

import java.io.Serializable;
import java.math.BigInteger;

/**
 * The base class of all tensors. A tensor is a multi-dimensional array.
 *
 * @param <T> Type of this tensor.
 * @param <U> Storage for entries of this tensor.
 * @param <V> Type (or wrapper) of an element of this tensor.
 */
public abstract class TensorBase<T extends TensorBase<T, U, V>, U, V>
        implements Serializable,
        TensorBinaryOpsMixin<T, T>,
        TensorUnaryOpsMixin<T>,
        TensorPropertiesMixin<V> {

    /**
     * Entry data of this tensor. If this tensor is dense, then this specifies all entris within this tensor. If this tensor is
     * sparse, this specifies only the non-zero entries of this tensor.
     */
    public final U entries;
    /**
     * The shape of this tensor.
     */
    public final Shape shape;


    /**
     * Creates a tensor with the specified entries and shape.
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     *                If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected TensorBase(Shape shape, U entries) {
        this.shape = shape;
        this.entries = entries;
        shape.makeStridesIfNull();  // TODO: investigate ways only compute strides if needed (e.g. matrices likly dont need strides).
    }


    /**
     * Gets the shape of this tensor.
     * @return The shape of this tensor.
     */
    @Override
    public Shape getShape() {
        return this.shape;
    }

    /**
     * Gets the element of this tensor at the specified indices.
     * @param indices Indices of the element to get.
     * @return The element of this tensor at the specified indices.
     * @throws IllegalArgumentException If {@code indices} is not of length 2.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    public abstract V get(int... indices);


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
     * Gets the entry data of this tensor as a 1D array.
     * @return The entries of this tensor.
     */
    public U getEntries() {
        return this.entries;
    }


    /**
     * Gets the total number of entries in this tensor.
     * @return The total number of entries in this tensor.
     */
    public BigInteger totalEntries() {
        return shape.totalEntries();
    }


    /**
     * Checks if a tensor has the same shape as this tensor.
     * @param B Second tensor.
     * @return True if this tensor and B have the same shape. False otherwise.
     */
    public boolean sameShape(TensorBase<?, ?, ?> B) {
        return this.shape.equals(B.shape);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     * @see #flatten(int)
     */
    public abstract T flatten();


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    public abstract T flatten(int axis);
}
