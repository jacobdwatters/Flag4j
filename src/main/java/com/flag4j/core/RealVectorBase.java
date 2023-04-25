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
import com.flag4j.Vector;

public abstract class RealVectorBase<T, W> extends
        VectorBase<T, Vector, W, T, double[], Double> {

    /**
     * Constructs a basic vector with the specified number of entries.
     *
     * @param size        Number of entries in this vector.
     * @param entries     The non-zero entries of this sparse tensor.
     */
    public RealVectorBase(int size, double[] entries) {
        super(size, entries);
    }


    /**
     * Constructs a real vector with the specified number of entries.
     *
     * @param shape   Number of entries in this vector.
     * @param entries The non-zero entries of this sparse tensor.
     * @throws IllegalArgumentException If the rank of the shape is not 1.
     */
    public RealVectorBase(Shape shape, double[] entries) {
        super(shape, entries);
    }
}