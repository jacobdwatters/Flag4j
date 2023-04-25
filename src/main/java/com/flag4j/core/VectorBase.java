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

import com.flag4j.Shape;
import com.flag4j.util.ParameterChecks;

/**
 * <p>
 *     The base class for all vectors. A vector is equivalent to a {@link TensorBase tensor} of rank 1. Vectors do not
 *     have an orientation (that is, row/column vector). However, several methods assume or allow the user to specify the
 *     orientation of the vector.
 * </p>
 * @param <T> Type of this vector.
 * @param <U> Dense vector type.
 * @param <V> Sparse Vector type.
 * @param <W> Complex vector type.
 * @param <Y> Real vector type.
 * @param <D> Type of the storage data structure for the vector.
 *           This common use case will be an array or list-like data structure.
 * @param <X> The type of individual entry within the {@code D} data structure.
 * @param <TT> The matrix type equivalent to this vector.
 * @param <UU> Dense Matrix type equivalent.
 * @param <WW> Complex matrix type equivalent.
 */
public abstract class VectorBase<T, U, V, W, Y, D, X extends Number,
        TT, UU, WW>
        extends TensorBase<T, U, W, Y, D, X>
        implements VectorPropertiesMixin,
        VectorManipulationsMixin<TT>,
        VectorComparisonsMixin,
        VectorOperationsMixin<T, U, V, W, X, TT, UU, WW> {

    /**
     * Size of the vector. This includes zero-entries if the vector is sparse.
     */
    public final int size;


    /**
     * Constructs a basic vector with the specified number of entries.
     *
     * @param size        Number of entries in this vector.
     * @param entries     The non-zero entries of this sparse tensor.
     */
    public VectorBase(int size, D entries) {
        super(new Shape(size), entries);
        this.size = size;
    }


    /**
     * Constructs a basic vector with the specified number of entries.
     *
     * @param shape        Number of entries in this vector.
     * @param entries     The non-zero entries of this sparse tensor.
     * @throws IllegalArgumentException If the rank of the shape is not 1.
     */
    public VectorBase(Shape shape, D entries) {
        super(shape, entries);
        ParameterChecks.assertRank(1, shape); // Ensure the shape is of rank 1.
        this.size = shape.get(0);
    }


    /**
     * Gets the size of this vector.
     *
     * @return The size, i.e. number of entries, of this vector.
     */
    public int size() {
        return size;
    }


    /**
     * Checks if a vector has the same number of elements as this vector. Same as {@link #sameShape(VectorBase)}
     * @param b Vector to compare to this vector.
     * @return True if this vector and {@code b} have the same number of elements.
     */
    public boolean sameSize(VectorBase<?, ?, ?, ?, ?, ?, ?, ?, ?, ?> b) {
        return this.size==b.size;
    }


    /**
     * Checks if a vector has the same number of elements as this vector. Same as {@link #sameSize(VectorBase)}
     * @param b Vector to compare to this vector.
     * @return True if this vector and {@code b} have the same number of elements.
     */
    public boolean sameShape(VectorBase<?, ?, ?, ?, ?, ?, ?, ?, ?, ?> b) {
        return this.size==b.size;
    }
}
