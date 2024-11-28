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

package org.flag4j.arrays.backend;


import org.flag4j.arrays.Shape;

import java.io.Serializable;
import java.math.BigInteger;

/**
 * <p>The base class of all tensors. A tensor is a multidimensional array which consists of:
 * <ul>
 *     <li>The {@link #shape} of the tensor. This specified the dimension of the tensor along each axes.
 *     The number of axes in this tensor is referred to as the "{@link #rank}" of the tensor and corresponds to the number of
 *     indices required to uniquely identify an element within the </li>
 *     <li>A one-dimensional container for the {@link #data} of the tensor. If the tensor is dense, this will contain all
 *     data of the tensor. If the tensor is sparse this will only contains the non-zero elements of the tensor.</li>
 * </ul>
 * 
 *
 * @param <T> Type of this tensor.
 * @param <U> Storage for data of this tensor.
 * @param <V> Type (or wrapper) of an element of this tensor.
 */
public abstract class AbstractTensor<T extends AbstractTensor<T, U, V>, U, V>
        implements Serializable, TensorMixin<T, U, V> {

    /**
     * Entry data of this tensor. If this tensor is dense, then this specifies all data within this tensor. If this tensor is
     * sparse, this specifies only the non-zero data of this tensor.
     */
    public final U data;
    /**
     * The shape of this tensor.
     */
    public final Shape shape;
    /**
     * The rank of this tensor. That is, the number of indices required to uniquely specify an element in the tensor (the number of
     * axes within this tensor).
     */
    public final int rank;


    /**
     * Creates a tensor with the specified data and shape.
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     *                If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractTensor(Shape shape, U data) {
        this.shape = shape;
        this.data = data;
        rank = shape.getRank();
    }


    /**
     * Gets the shape of this tensor.
     * @return The shape of this tensor.
     */
    @Override
    public Shape getShape() {
        return shape;
    }


    /**
     * Gets the element of this tensor at the specified indices.
     * @param indices Indices of the element to get.
     * @return The element of this tensor at the specified indices.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    @Override
    public abstract V get(int... indices);


    /**
     * Sets the element of this tensor at the specified indices.
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    public abstract T set(V value, int... indices);


    /**
     * <p>Gets the rank of this tensor. That is, number of indices needed to uniquely select an element of the tensor. This is also te
     * number of dimensions (i.e. order/degree) of the tensor.
     *
     * <p>Note, this method is distinct from the {@code matrix rank}.
     *
     * @return The rank of this tensor.
     */
    @Override
    public int getRank() {
        return rank;
    }


    /**
     * Gets the entry data of this tensor as a 1D array.
     * @return The data of this tensor.
     */
    @Override
    public U getData() {
        return data;
    }


    /**
     * Gets the total number of data in this tensor.
     * @return The total number of data in this tensor.
     */
    public BigInteger totalEntries() {
        return shape.totalEntries();
    }


    /**
     * Checks if a tensor has the same shape as this tensor.
     * @param B Second tensor.
     * @return True if this tensor and B have the same shape. False otherwise.
     */
    public boolean sameShape(AbstractTensor<?, ?, ?> B) {
        return shape.equals(B.shape);
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
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


    /**
     * Copies and reshapes this tensor.
     * @param newShape New shape for the tensor.
     * @return A copy of this tensor with the new shape.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    public abstract T reshape(Shape newShape);


    /**
     * Copies and reshapes this tensor.
     * @param dims The dimensions of the new shape.
     * @return A copy of this tensor with the new shape.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code dims} does not represent a shape broadcastable to
     * {@link #shape this.shape}.
     */
    public T reshape(int... dims) {
        return reshape(new Shape(dims));
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    public abstract T makeLikeTensor(Shape shape, U entries);
}
