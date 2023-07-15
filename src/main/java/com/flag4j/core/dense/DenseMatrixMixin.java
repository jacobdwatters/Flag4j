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

package com.flag4j.core.dense;

import com.flag4j.Matrix;

/**
 * This interface specifies methods which all dense matrices should implement.
 * @param <T> Type of the dense matrix.
 * @param <X> Type of an individual entry in the matrix.
 */
public interface DenseMatrixMixin<T, X> {


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    T setValues(X[][] values);


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    T setValues(Double[][] values);


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    T setValues(double[][] values);


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    T setValues(int[][] values);


    /**
     * Computes the element-wise addition of a matrix with a real dense matrix. The result is stored in this matrix.
     * @param B The matrix to add to this matrix.
     */
    void addEq(Matrix B);


    /**
     * Computes the element-wise subtraction of this matrix with a real dense matrix. The result is stored in this matrix.
     * @param B The matrix to subtract from this matrix.
     */
    void subEq(Matrix B);
}
