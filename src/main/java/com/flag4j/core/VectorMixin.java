/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package com.flag4j.core;


/**
 * This interface specifies methods which all vectors should implement.
 * @param <T> Type of this vector.
 * @param <U> Dense vector type.
 * @param <V> Sparse Vector type.
 * @param <W> Complex vector type.
 * @param <X> The type of individual entry within the vector.
 * @param <TT> The matrix type equivalent to this vector.
 * @param <UU> Dense Matrix type equivalent.
 * @param <WW> Complex matrix type equivalent.
 */
public interface VectorMixin<T, U, V, W, X extends Number, TT, UU, WW>
        extends VectorPropertiesMixin,
        VectorManipulationsMixin<TT>,
        VectorComparisonsMixin,
        VectorOperationsMixin<T, U, V, W, X, TT, UU, WW> {

    // TODO: Add default methods for methods which are the same for all vectors.

    /**
     * gets the size of this vector.
     * @return
     */
    int size();
}
