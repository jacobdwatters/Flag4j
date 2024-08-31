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

package org.flag4j.core_temp.arrays.sparse;

import org.flag4j.core_temp.MatrixMixin;
import org.flag4j.core_temp.VectorMixin;
import org.flag4j.core_temp.arrays.dense.DenseVectorMixin;

/**
 * This interface specifies methods which all sparse vectors should implement.
 * @param <T> Type of this sparse vector.
 * @param <U> Type of equivalent dense vector.
 * @param <V> Type of matrix equivalent to {@code T}.
 * @param <W> Type of dense matrix equivalent to {@code V}.
 * @param <Y> Type (or wrapper of) an individual element in this vector.
 */
public interface SparseVectorMixin<T extends SparseVectorMixin<T, U, V, W, Y>, U extends DenseVectorMixin<U, T, ?, Y>,
        V extends MatrixMixin<V, ?, Y>, W extends MatrixMixin<W, ?, Y>, Y>
        extends VectorMixin<T, V, W, Y>, SparseTensorMixin<U, T> {
}
