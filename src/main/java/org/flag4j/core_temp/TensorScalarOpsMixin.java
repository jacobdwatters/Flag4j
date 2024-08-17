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

import org.flag4j.core_temp.structures.fields.Field;


/**
 * This interface defines scalar operations_old between a tensor and a scalar field element.
 * @param <T> Type of the tensor returned by the operation.
 * @param <U> Type of the scalar field element.
 * @see TensorPrimitiveOpsMixin
 */
public interface TensorScalarOpsMixin<T extends TensorBase<T, ?, ?>, U extends Field<U>> {

    /**
     * Adds a scalar value to each element of this tensor.
     * @param a Value to add to each entry of this tensor.
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    public T add(U a);


    /**
     * Subtracts a scalar value from each element of this tensor.
     * @param a Value to subtract from each entry of this tensor.
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    public T sub(U a);


    /**
     * Computes the sclar multiplication between this tensor and the specified scalar {@code factor}.
     * @param factor Scalar factor to apply to this tensor.
     * @return The sclar product of this tensor and {@code factor}.
     */
    public T mult(U factor);


    /**
     * Computes the scalar division of this tensor and the specified scalar {@code factor}.
     * @param divisor The scalar value to divide this tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     */
    public T div(U divisor);


    /**
     * Sums all elements of this tensor.
     * @return The sum of all elements of this tensor.
     */
    public U sum();


    /**
     * Computes the product all elements of this tensor.
     * @return The product of all elements of this tensor.
     */
    public U prod();
}
