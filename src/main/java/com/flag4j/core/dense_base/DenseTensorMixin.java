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

package com.flag4j.core.dense_base;


import com.flag4j.dense.Tensor;

/**
 * This interface specifies methods which should be implemented by all dense tensors.
 */
public interface DenseTensorMixin {

    /**
     * Computes the element-wise addition between this tensor and the specified tensor and stores the result
     * in this tensor.
     * @param B tensor to add to this tensor.
     * @throws IllegalArgumentException If this tensor and the specified tensor have different .
     */
    void addEq(Tensor B);


    /**
     * Computes the element-wise subtraction between this tensor and the specified tensor and stores the result
     * in this tensor.
     * @param B tensor to subtract this tensor.
     * @throws IllegalArgumentException If this tensor and the specified tensor have different shapes.
     */
    void subEq(Tensor B);
}
