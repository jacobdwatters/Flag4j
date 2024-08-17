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
import org.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations_old which all tensors (i.e. matrices and vectors) should implement.
 *
 * @param <T> TensorOld type.
 * @param <U> Dense tensor type.
 * @param <W> Complex tensor type.
 * @param <Z> Dense complex tensor type.
 * @param <Y> Real tensor type.
 * @param <X> TensorOld entry type.
 */
interface TensorOperationsMixin<T, U, W, Z, Y, X extends Number> {

    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    T add(T B);


    /**
     * Adds specified value to all entries of this tensor.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    U add(double a);


    /**
     * Adds specified value to all entries of this tensor.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    Z add(CNumber a);


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    T sub(T B);


    /**
     * Adds specified value to all entries of this tensor.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    U sub(double a);


    /**
     * Subtracts a specified value from all entries of this tensor.
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    Z sub(CNumber a);


    /**
     * Computes scalar multiplication of a tensor.
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    T mult(double factor);


    /**
     * Computes scalar multiplication of a tensor.
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    W mult(CNumber factor);


    /**
     * Computes the scalar division of a tensor.
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    T div(double divisor);


    /**
     * Computes the scalar division of a tensor.
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    W div(CNumber divisor);


    /**
     * Sums together all entries in the tensor.
     * @return The sum of all entries in this tensor.
     */
    X sum();


    /**
     * Computes the element-wise square root of a tensor.
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    T sqrt();


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    Y abs();


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     * @return The transpose of this tensor.
     */
    default T transpose(){
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * @return The transpose of this tensor.
     */
    T T();


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    T recip();


    /**
     * Gets the element in this tensor at the specified indices.
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    X get(int... indices);


    /**
     * Creates a copy of this tensor.
     * @return A copy of this tensor.
     */
    T copy();


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    T elemMult(T B);


    /**
     * Computes the element-wise division between two tensors.
     * @param B TensorOld to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    T elemDiv(U B);
}
