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

package org.flag4j.arrays.backend;


/**
 * This interface specified methods which all dense vectors should implement.
 * @param <T> Type of this dense vector.
 * @param <U> Type of equivalent sparse vector.
 * @param <V> Type of matrix equivalent to {@code T}.
 * @param <Y> Type (or wrapper of) an individual element in this vector.
 */
public interface DenseVectorMixin<T extends DenseVectorMixin<T, U, V, W, Y>, U extends SparseVectorMixin<U, T, ?, V, W, Y>,
        V extends DenseMatrixMixinOld<V, ?, T, W, Y>, W, Y>
        extends VectorMixinOld<T, V, V, W, Y>, DenseTensorMixin<T, U> {


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and {@code b}.
     * @throws IllegalArgumentException If either this vector or {@code b} do not have exactly 3 entries.
     */
    public T cross(T b);
}
