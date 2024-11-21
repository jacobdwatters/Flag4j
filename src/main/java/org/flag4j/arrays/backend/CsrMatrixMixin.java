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
 * This interface specifies methods which all sparse CSR matrices should implement.
 * @param <T> Type of this sparse CSR matrix.
 * @param <U> Type of dense matrix equivalent to {@code T}.
 * @param <V> Type of vector similar to {@code T}.
 * @param <W> Type of dense vector equivalent to {@code V}.
 * @param <Y> Type (or wrapper of) of an element of this matrix.
 */
public interface CsrMatrixMixin<T extends CsrMatrixMixin<T, U, V, W, X, Y>,
        U extends DenseMatrixMixinOld<U, ?, W, X, Y>,
        V extends SparseVectorMixin<V, W, ?, U, X, Y>,
        W extends DenseVectorMixin<W, V, U, X, Y>, X, Y>
        extends MatrixMixinOld<T, U, V, W, X, Y> {

    /**
     * The density of this sparse CSR matrix. That is, the decimal percentage of elements in this matrix which are non-zero.
     * @return The density of this sparse matrix.
     */
    public default double density() {
        return 1.0 - sparsity();
    }


    /**
     * The sparsity of this sparse CSR matrix. That is, the decimal percentage of elements in this matrix which are zero.
     * @return The density of this sparse matrix.
     */
    public double sparsity();


    /**
     * Converts this sparse matrix to an equivalent dense matrix.
     * @return A dense matrix equivalent to this sparse CSR matrix.
     */
    public U toDense();


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices();
}
