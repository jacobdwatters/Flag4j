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

import org.flag4j.core_old.MatrixPropertiesMixin;

/**
 * This interface specifies methods which all tensors should implement.
 *
 * @param <T> Type of this tensor.
 * @param <U> Type (or wrapper) of an element of this tensor.
 */
public interface TensorMixin<T extends TensorMixin<T, U>, U>
        extends TensorBinaryOpsMixin<T, T>,
        TensorUnaryOpsMixin<T>,
        TensorPropertiesMixin<U>{

    /**
     * Gets the element of this tensor at the specified indices.
     * @param indices Indices of the element to get.
     * @return The element of this tensor at the specified indices.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    public U get(int... indices);


    /**
     * <p>
     * Gets the rank of this tensor. That is, number of indices needed to uniquely select an element of the tensor. This is also te
     * number of dimensions (i.e. order/degree) of the tensor.
     * </p>
     *
     * <p>
     * Note, this method is distinct from the {@link MatrixPropertiesMixin#matrixRank()} method.
     * </p>
     *
     * @return The rank of this tensor.
     */
    public int getRank();
}
