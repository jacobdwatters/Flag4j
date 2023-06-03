/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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
 * This interface specifies methods which all real tensors should implement.
 * @param <T> Tensor type.
 * @param <W> Complex tensor type.
 */
public interface RealTensorMixin<T, W> {


    /**
     * Checks if this tensor contains only non-negative values.
     * @return True if this tensor only contains non-negative values. Otherwise, returns false.
     */
    boolean isPos();


    /**
     * Checks if this tensor contains only non-positive values.
     * @return trie if this tensor only contains non-positive values. Otherwise, returns false.
     */
    boolean isNeg();


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link com.flag4j.complex_numbers.CNumber CNumber} rather than {@link Double}.
     * @return A complex matrix which is equivalent to this matrix.
     */
    W toComplex();
}
