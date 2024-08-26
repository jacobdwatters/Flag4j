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

package org.flag4j.core_temp;


import org.flag4j.core.Shape;

/**
 * <p>This abstract class defines a tensor whose elements satisfy the axioms of a ring.</p>
 *
 * <p>To allow for primitive types, the elements of this tensor do not neccesarily have to implement
 * {@link org.flag4j.core_temp.structures.rings.Ring}.</p>
 *
 * <p>Formally, an ring is a set <b>R</b> with the binary operations_old addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>R</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition is commutative: a + b = b + a</li>
 *      <li>Existince of additive and multiplicitive identities: There exisits two distinct elements 0 and 1 in <b>R</b> sucht that a + 0 = 0
 *      and a * 1 = 1 (called the addative and multiplicitive identities respectively).</li>
 *      <li>Existince of addative inverse: There exists an element -a in <b>R</b> such that a + (-a) = 0.</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * </p>
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of a dense tensor equivalent to {@code T}. If {@code T} is dense, then this should be the same type as {@code T}.
 * This parameter is required because some operations (e.g. {@link #tensorDot(TensorOverRing, int)}) between two sparse tensors
 * result in a dense tensor.
 * @param <V> Storage for entries of this tensor.
 * @param <W> Type (or wrapper) of an element of this tensor. Should satisfy the axioms of a ring as stated.
 */
public abstract class TensorOverRing<T extends TensorOverRing<T, U, V, W>,
        U extends TensorOverRing<U, U, V, W>, V, W> extends TensorOverSemiRing<T, U, V, W> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected TensorOverRing(Shape shape, V entries) {
        super(shape, entries);
    }


    /**
     * Subtracts a sclar value from each entry of this tensor.
     *
     * @param b Scalar value in differencce.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    public abstract T sub(W b);


    /**
     * Subtracts a sclar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in differencce.
     */
    public abstract void subEq(W b);


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws org.flag4j.util.exceptions.TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public abstract T sub(T b);


    /**
     * Computes the element-wise absolute value of this tensor.
     * @return The element-wise absolute value of this tensor.
     */
    public abstract TensorOverRing abs();


    /**
     * Computes the element-wise conjugation of this tensor.
     * @return The element-wise conjugation of this tensor.
     */
    public abstract T conj();


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     * @return The conjugate transpose of this tensor.
     * @see #H(int, int)
     * @see #H(int...)
     */
    public abstract T H();


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     * @return The conjugate transpose of this tensor acording to the specified axes.
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    public abstract T H(int axis1, int axis2);


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    public abstract T H(int... axes);
}
