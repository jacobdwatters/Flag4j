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
 */
public interface DenseMatrixMixin {

    /**
     * Computes the element-wise addition between this matrix and the specified matrix and stores the result
     * in this matrix.
     * @param B matrix to add to this matrix.
     * @throws IllegalArgumentException If this matrix and the specified matrix have different .
     */
    void addEq(Matrix B);


    /**
     * Computes the element-wise subtraction between this matrix and the specified matrix and stores the result
     * in this matrix.
     * @param B matrix to subtract this matrix.
     * @throws IllegalArgumentException If this matrix and the specified matrix have different shapes.
     */
    void subEq(Matrix B);
}
