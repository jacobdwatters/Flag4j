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


/**
 * This tensor specified operations_old between tensors and primitive values.
 *
 * @param <T> TensorOld type that the operation returns.
 * @see TensorPrimitiveOpsMixin
 */
public interface TensorPrimitiveOpsMixin<T extends TensorOverField<T, ?, ?>> {

    /**
     * Adds a scalar value to each element of this tensor.
     * @param b Value to add to each entry of this tensor.
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    public T add(double b);


    /**
     * Subtracts a scalar value from each element of this tensor.
     * @param b Value to subtract from each entry of this tensor.
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    public T sub(double b);


    /**
     * Computes the sclar multiplication between this tensor and the specified scalar {@code factor}.
     * @param factor Scalar factor to apply to this tensor.
     * @return The sclar product of this tensor and {@code factor}.
     */
    public T mult(double factor);


    /**
     * Computes the scalar division of this tensor and the specified scalar {@code factor}.
     * @param divisor The scalar value to divide this tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     */
    T div(double divisor);
}
