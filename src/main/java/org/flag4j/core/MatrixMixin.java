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

package org.flag4j.core;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;

/**
 * This interface specified methods which all matrices should implement.
 * @param <T> Matrix type.
 * @param <U> Dense matrix type.
 * @param <V> Sparse matrix type.
 * @param <W> Complex matrix type.
 * @param <VV> Complex sparse matrix type.
 * @param <Y> Real matrix type.
 * @param <X> Matrix entry type.
 * @param <TT> Vector type equivalent.
 * @param <UU> Dense vector type.
 */
public interface MatrixMixin<
        T,
        U, V, W, VV, X extends Number,
        TT extends VectorMixin<TT, UU, ?, ?, ?, ?, ?, ?>,
        UU extends VectorMixin<UU, UU, ?, CVector, X, U, U, CMatrix>>
        extends MatrixPropertiesMixin,
        MatrixComparisonsMixin<T>,
        MatrixManipulationsMixin<T, X>,
        MatrixOperationsMixin<T, U, V, W, VV, X, TT, UU> {

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
    X get(int... indices); // This method is specified here and in the TensorMixin interface intentionally.


    /**
     * Gets the shape of this matrix.
     * @return The shape of this matrix.
     */
    Shape shape();
}


