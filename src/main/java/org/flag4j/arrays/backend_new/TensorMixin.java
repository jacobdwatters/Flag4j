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

package org.flag4j.arrays.backend_new;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseMatrixMixinOld;
import org.flag4j.arrays.backend.TensorBinaryOpsMixin;

/**
 * This interface specifies methods which all tensors should implement.
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of storage for the entries of this tensor. Should be array-like.
 * @param <V> Type (or wrapper) of an element of this tensor.
 */
public interface TensorMixin<T extends TensorMixin<T, U, V>, U, V>
        extends TensorBinaryOpsMixin<T, T> {

    /**
     * Gets the entries of this tensor.
     * @return The entries of this tensor.
     */
    U getEntries();


    /**
     * Gets the element of this tensor at the specified indices.
     * @param indices Indices of the element to get.
     * @return The element of this tensor at the specified indices.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    V get(int... indices);


    /**
     * <p>
     * Gets the rank of this tensor. That is, number of indices needed to uniquely select an element of the tensor. This is also te
     * number of dimensions (i.e. order/degree) of the tensor.
     * </p>
     *
     * <p>
     * Note, this method is distinct from the {@link DenseMatrixMixinOld#matrixRank()} method.
     * </p>
     *
     * @return The rank of this tensor.
     */
    int getRank();


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     * @return The transpose of this tensor.
     * @see #T(int, int)
     * @see #T(int...)
     */
    default T T() {
        return T(0, getRank());
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor according to the specified axes.
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    T T(int axis1, int axis2);


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    T T(int... axes);


    /**
     * Creates a deep copy of this tensor.
     * @return A deep copy of this tensor.
     */
    T copy();


    /**
     * Gets the shape of this tensor.
     * @return The shape of this tensor.
     */
    Shape getShape();
}
