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


import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies comparisons which all tensors (i.e. matrices and vectors) should implement.
 *
 * @param <T> Tensor type.
 * @param <U> Dense Tensor type.
 * @param <V> Sparse Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Y> Real Tensor type.
 * @param <X> Tensor entry type.
 */
public interface TensorComparisonsMixin<T extends TensorBase<?>, U extends TensorBase<?>, V extends TensorBase<?>,
        W extends TensorBase<CNumber[]>, Y extends TensorBase<double[]>, X extends Number> {


    /**
     * Checks if this tensor only contains zeros.
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    boolean isZeros();


    /**
     * Checks if this tensor only contains ones.
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    boolean isOnes();


    /**
     * Checks if this tensor is equal to a specified Object.
     * @param B Object to compare this tensor to.
     */
    @Override
    boolean equals(Object B);
}
