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


import org.flag4j.algebraic_structures.rings.SemiRing;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ArrayUtils;


/**
 * <p>This abstract class defines a tensor whose elements satisfy the axioms of a semi-ring.</p>
 *
 * <p>To allow for primitive types, the elements of this tensor do not necessarily have to implement
 * {@link SemiRing}.</p>
 *
 * <p>Formally, an semi-ring is a set <b>R</b> with the binary operations_old addition (+) and multiplication (*)
 * defined such that for elements a, b, c in <b>R</b> the following are satisfied:
 *  <ul>
 *      <li>Addition and multiplication are associative: a + (b + c) = (a + b) + c and a * (b * c) = (a * b) * c.</li>
 *      <li>Addition is commutative: a + b = b + a</li>
 *      <li>Existence of additive and multiplicative identities: There exists two distinct elements 0 and 1 in <b>R</b> such that a + 0 = 0
 *      and a * 1 = 1 (called the additive and multiplicative identities respectively).</li>
 *      <li>Distributivity of multiplication over addition: a * (b + c) = (a * b) + (a * c).</li>
 *  </ul>
 * </p>
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. If {@code T} is dense, then this should be the same type as {@code T}.
 * This parameter required because some operations (e.g. {@link #tensorDot(TensorOverSemiRing, int)}) between two sparse tensors
 * result in
 * a dense tensor.
 * @param <V> Storage for entries of this tensor.
 * @param <W> Type (or wrapper) of an element of this tensor. Should satisfy the axioms of a semi-ring as stated.
 */
public abstract class TensorOverSemiRing<T extends TensorOverSemiRing<T, U, V, W>,
        U extends TensorOverSemiRing<U, U, V, W>, V, W> extends TensorBase<T, V, W> {


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected TensorOverSemiRing(Shape shape, V entries) {
        super(shape, entries);
    }


    /**
     * Adds a scalar value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    public abstract T add(W b);


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    public abstract void addEq(W b);


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     * @param b Second tensor in the element-wise sum.
     * @return The sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public abstract T add(T b);


    /**
     * Multiplies a scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    public abstract T mult(W b);


    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    public abstract void multEq(W b);


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     * @param b Second tensor in the element-wise product.
     * @return The element-wise product between this tensor and {@code b}.
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public abstract T elemMult(T b);


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. If {@code axes=N}, then
     * the product sums will be computed along the last {@code N} dimensions of this tensor and the first {@code N} dimensions of
     * the {@code src2} tensor.
     * @param src2 TensorOld to contract with this tensor.
     * @param axes Axes specifying the number of axes to compute the tensor dot product over. If {@code axes=N}, then
     *             the product sums will be computed along the last {@code N} dimensions of this tensor and the first {@code N}
     *             dimensions of.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     * and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    public U tensorDot(T src2, int axes){
        int rank2 = src2.getRank();
        int[] src1Axes = ArrayUtils.intRange(0, axes);
        int[] src2Axes = ArrayUtils.intRange(rank2-axes, rank2);

        return tensorDot(src2, src1Axes, src2Axes);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. That is,
     * computes the sum of products between the two tensors along the specified axes.
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxis Axis along which to compute products for this tensor.
     * @param bAxis Axis along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     * and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    public U tensorDot(T src2, int aAxis, int bAxis) {
        return tensorDot(src2, new int[]{aAxis}, new int[]{bAxis});
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     * {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor.
     */
    public abstract U tensorDot(T src2, int[] aAxes, int[] bAxes);


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     * @param src2 TensorOld to compute dot product with this tensor.
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     * along the second-to-last axis.
     */
    public U tensorDot(T src2) {
        return tensorDot(src2, getRank()-1, getRank()-2);
    }


    /**
     * <p>Computes the generalized tensor trace of this tensor along first and second axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by the
     * first and
     * second axes. The shape of the resulting tensor is equal to this tensor with the first and second axes removed.</p>
     *
     * @return The generalized trace of this tensor along first and second axes.
     */
    public T tensorTr() {
        return tensorTr(0, 1);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     * (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    public abstract T tensorTr(int axis1, int axis2);


    /**
     * Checks if this tensor only contains zeros.
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    public abstract boolean isZeros();


    /**
     * Computes the sum of all values in this tensor.
     * @return The sum of all values in this tensor.
     */
    public abstract W sum();


    /**
     * Computes the product of all values in this tensor.
     * @return The product of all values in this tensor.
     */
    public abstract W prod();
}
