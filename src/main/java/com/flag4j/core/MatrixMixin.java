/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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


import com.flag4j.Shape;

/**
 * This interface specified methods which all matrices should implement.
 * @param <T> Matrix type.
 * @param <U> Dense Matrix type.
 * @param <V> Sparse Matrix type.
 * @param <W> Complex Matrix type.
 * @param <Y> Real Matrix type.
 * @param <X> Matrix entry type.
 * @param <TT> Vector type equivalent.
 * @param <UU> Dense vector type.
 */
public interface MatrixMixin<T, U, V, W, X extends Number,
        TT extends VectorMixin<TT, UU, ?, ?, X, T, U, W>,
        UU extends VectorMixin<UU, UU, ?, ?, X, U, U, W>>
        extends MatrixPropertiesMixin,
        MatrixComparisonsMixin<T>,
        MatrixManipulationsMixin<T, X>,
        MatrixOperationsMixin<T, U, V, W, X, TT, UU> {

    /**
     * Gets the number of rows in this matrix.
     * @return The number of rows in this matrix.
     */
    int numRows();


    /**
     * Gets the number of columns in this matrix.
     * @return The number of columns in this matrix.
     */
    int numCols();


    /**
     * Gets the element of this matrix at the specified indices.
     * @param indices Indices of the element to get.
     * @return The element of this matrix at the specified indices.
     * @throws IllegalArgumentException If {@code indices} is not of length 2.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    X get(int... indices); // This method is specified here an in the tensor mixin interface intentionally.


    /**
     * Gets the shape of this matrix.
     * @return The shape of this matrix.
     */
    Shape shape();
}


