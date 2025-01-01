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

package org.flag4j.arrays.backend.ring_arrays;


import org.flag4j.algebraic_structures.Ring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringMatrix;
import org.flag4j.linalg.ops.dense.ring_ops.DenseRingTensorOps;
import org.flag4j.util.exceptions.TensorShapeException;

/**
 * The base class for all dense matrices whose elements are members of a {@link Ring}.
 * @param <T> The type of this matrix.
 * @param <U> The type of the vector which is of similar type to {@link T &lt;T&gt;}.
 * @param <V> The type of the arrays the data of the matrix belong to.
 */
public abstract class AbstractDenseRingMatrix<T extends AbstractDenseRingMatrix<T, U, V>,
        U extends AbstractDenseRingVector<U, T, V>, V extends Ring<V>>
        extends AbstractDenseSemiringMatrix<T, U, V>
        implements RingTensorMixin<T, T, V>, MatrixMixin<T, T, U, V> {


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseRingMatrix(Shape shape, V[] data) {
        super(shape, data);
    }


    /**
     * Computes the element-wise difference of two matrices.
     *
     * @param b Second matrix in the element-wise difference.
     *
     * @return The element-wise difference of this matrix and {@code b}.
     */
    @Override
    public T sub(T b) {
        V[] dest = makeEmptyDataArray(data.length);
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Computes the element-wise difference between two matrices of the same shape and stores the result in this matrix.
     *
     * @param b Second matrix in the element-wise difference.
     *
     * @throws TensorShapeException If this matrix and {@code b} do not have the same shape.
     */
    public void subEq(T b) {
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, data);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        return T();
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public T H(int... axes) {
        return T(axes);
    }
}
