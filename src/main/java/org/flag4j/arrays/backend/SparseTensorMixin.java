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

/**
 * This interface specifies methods which sparse tensors should implement.
 * @param <T> The equivalent dense tensor type.
 * @param <U> The type of this sparse tensor.
 */
public interface SparseTensorMixin<T extends DenseTensorMixin<T, U>, U extends SparseTensorMixin<T, U>> {


    /**
     * The density of this sparse tensor. That is, the percentage of elements in this tensor which are non-zero as a decimal.
     * @return The density of this sparse tensor.
     */
    public default double density() {
        return 1.0 - sparsity();
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     * @return The density of this sparse tensor.
     */
    public double sparsity();


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     * @return A dense tensor equivalent to this sparse tensor.
     */
    public T toDense();


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices();
}
