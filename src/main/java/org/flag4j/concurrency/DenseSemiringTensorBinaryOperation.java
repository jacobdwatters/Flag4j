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

package org.flag4j.concurrency;

import org.flag4j.algebraic_structures.Semiring;
import org.flag4j.arrays.Shape;

/**
 * This interface specifies a binary operation on two dense {@link Semiring} tensors.
 */
public interface DenseSemiringTensorBinaryOperation {

    /**
     * Applies the specified binary operation on the two dense tensors.
     * @param src1 Entries of the first tensor.
     * @param shape1 Shape of the first tensor.
     * @param src2 Entries of the second tensor.
     * @param shape2 Shape of the second tensor.
     * @param dest Array to store the result of the binary operation of the two tensors in.
     */
    public <T extends Semiring<T>> void apply(T[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest);
}
