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


import com.flag4j.CTensor;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;

/**
 * This abstract class is the base class for all complex tensors.
 * @param <T> Type of this tensor.
 * @param <Y> Real tensor type.
 */
public abstract class ComplexTensorBase<T, Y>
        extends TensorBase<T, CTensor, T, Y, CNumber[], CNumber>
        implements ComplexTensorMixin<T, Y> {

    /**
     * Creates a complex tensor with specified entries and shape.
     *
     * @param shape   Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     */
    public ComplexTensorBase(Shape shape, CNumber[] entries) {
        super(shape, entries);
    }
}
