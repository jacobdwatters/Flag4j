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
import org.flag4j.core_temp.arrays.dense.DenseMatrixMixin;

/**
 * This interface specifies methods which all sparse COO matrices should implement.
 * @param <T> Type of this sparse matrix.
 * @param <V> Type of dense matrix which is equivalent to {@code T}.
 * @param <W> Type (or wrapper of) an individual element in the matrix.
 */
public interface CooMatrixMixin<T extends CooMatrixMixin<T, V, W>,
        V extends DenseMatrixMixin<V, T, ?, W>, W>
        extends MatrixMixin<T, V, W>, SparseTensorMixin<V, T> {

    // TODO: consider adding toCsr() here.
}
