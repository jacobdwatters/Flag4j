/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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
 * The base class for all vectors.
 * @param <T> The type of entries for this matrix.
 */
public abstract class VectorBase<T> extends TensorBase<T> {

    /**
     * The orientation of this vector.
     */
    VectorOrientation orientation;
    /**
     * Size of the matrix.
     */
    public final int size;

    /**
     * Constructs a basic vector with the specified number of entries.
     * @param size Number of entries in this vector.
     * @param orientation Orientation of this vector.
     * @param entries The non-zero entries of this sparse tensor.
     */
    public VectorBase(int size, VectorOrientation orientation, T entries) {
        super(new Shape(size), entries);
        this.orientation = orientation;
        this.size = size;
    }


    /**
     * Gets the size of this vector.
     * @return The size, i.e. number of entries, of this vector.
     */
    public int size() {
        return size;
    }


    /**
     * Gets the oriented shape of this vector.
     * @return The oriented shape of this vector. <br>
     * If this vector is a {@link VectorOrientation#ROW row} vector, then
     * the shape will be {@code (1, this.size())}. <br>
     * If this vector is a {@link VectorOrientation#COL column} vector or an {@link VectorOrientation#UNORIENTED unoriented}
     * vector then the shape will be {@code (1, this.size())}.
     */
    public Shape getOrientedShape() {
        Shape orientedShape;

        if(this.orientation== VectorOrientation.ROW) {
            orientedShape = new Shape(1, this.size());
        } else {
            orientedShape = new Shape(this.size(), 1);
        }

        return orientedShape;
    }


    /**
     * Gets the orientation of this vector.
     * @return The orientation of this vector.
     */
    public VectorOrientation getOrientation() {
        return this.orientation;
    }
}
