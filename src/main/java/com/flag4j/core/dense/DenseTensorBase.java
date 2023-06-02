/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.core.dense;
import com.flag4j.Shape;
import com.flag4j.core.TensorBase;

import java.io.Serializable;

/**
 * This is the base class for all dense tensors.
 * @param <T> Type of this tensor.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type.
 * @param <D> Type of the storage data structure for the tensor.
 *           The common use case will be an array or list-like data structure.
 * @param <X> The type of individual entry within the {@code D} data structure
 */
public abstract class DenseTensorBase<T, W, Y, D extends Serializable, X extends Number>
        extends TensorBase<T, T, W, W, Y, D, X>
        implements DenseMixin<X> {


    /**
     * Creates a dense tensor with specified entries and shape.
     *
     * @param shape   Shape of this tensor.
     * @param entries Entries of this tensor. The number of entries must match the product of
     *                all {@code shape} dimensions. <b>Warning</b>: this is not enforced.
     */
    public DenseTensorBase(Shape shape, D entries) {
        super(shape, entries);
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    protected abstract T makeTensor(Shape shape, D entries);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    abstract void addEq(T B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    abstract void subEq(T B);
}
