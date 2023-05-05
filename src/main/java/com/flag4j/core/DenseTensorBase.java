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

package com.flag4j.core;
import com.flag4j.Shape;

/**
 * This is the base class for all dense tensors.
 * @param <T> Type of this tensor.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type.
 * @param <D> Type of the storage data structure for the tensor.
 *           This common use case will be an array or list-like data structure.
 * @param <X> The type of individual entry within the {@code D} data structure
 */
public abstract class DenseTensorBase<T, W, Y, D, X extends Number>
        extends TensorBase<T, T, W, W, Y, D, X> {


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



    // TODO: add abstract methods, toSparse(), etc.
}
