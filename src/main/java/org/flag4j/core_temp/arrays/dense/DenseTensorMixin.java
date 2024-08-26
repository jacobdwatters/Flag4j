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

package org.flag4j.core_temp.arrays.dense;


import org.flag4j.core_temp.arrays.sparse.SparseTesnorMixin;

/**
 * This interface specifies methods which sparse tensors should implement.
 * @param <T> The equivalent dense tensor type.
 * @param <U> The type of this sparse tensor.
 */
public interface DenseTensorMixin<T extends DenseTensorMixin<T, U>, U extends SparseTesnorMixin<T, U>> {


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    public U toCoo();


    /**
     * Computes the element-wise multiplication of two tensors and stores the result in this tensor.
     * @param b Second tensor in the element-wise product.
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public void elemMultEq(T b);


    /**
     * Computes the element-wise sum between two tensors and stores the result in this tensors.
     * @param b Second tensor in the element-wise sum.
     * @throws org.flag4j.util.exceptions.TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void addEq(T b);


    /**
     * Computes the element-wise difference between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws org.flag4j.util.exceptions.TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void subEq(T b);


    /**
     * Computes the element-wise division between two tensors and stores the result in this tensor.
     * @param b The denominator tensor in the element-wise quotient.
     * @throws org.flag4j.util.exceptions.TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    public void divEq(T b);


    /**
     * Computes the element-wise division between two tensors.
     * @param b The denominator tensor in the element-wise quotient.
     * @return The element-wise quotient of this tensor and {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    public T div(T b);
}
