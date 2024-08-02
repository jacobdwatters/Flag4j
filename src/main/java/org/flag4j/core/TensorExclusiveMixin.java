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

package org.flag4j.core;

import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.util.ArrayUtils;


/**
 * This interface contains several methods which should be implemented by all tensors which are <b>NOT</b> a matrix or
 * vector.
 *
 * @param <T> Tensor type.
 * @param <U> Dense tensor type.
 * @param <V> Sparse tensor type.
 * @param <W> Complex tensor type.
 */
public interface TensorExclusiveMixin<
        T extends TensorBase<T, U, W, ?, ?, ?, ?>,
        U, V, W> {

    // TODO: add toVector() and toMatrix() methods.

    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. If {@code axes=N}, then
     * the product sums will be computed along the last {@code N} dimensions of this tensor and the first {@code N} dimensions of
     * the {@code src2} tensor.
     * @param src2 Tensor to contract with this tensor.
     * @param axes Axes specifying the number of axes to compute the tensor dot product over. If {@code axes=N}, then
     *             the product sums will be computed along the last {@code N} dimensions of this tensor and the first {@code N}
     *             dimensions of.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     * and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    default T tensorDot(T src2, int axes){
        int rank2 = src2.getRank();
        int[] src1Axes = ArrayUtils.intRange(0, axes);
        int[] src2Axes = ArrayUtils.intRange(rank2-axes, rank2);

        return tensorDot(src2, src1Axes, src2Axes);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. That is,
     * computes the sum of products between the two tensors along the specified axes.
     * @param src2 Tensor to contract with this tensor.
     * @param aAxis Axis along which to compute products for this tensor.
     * @param bAxis Axis along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     * and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    default T tensorDot(T src2, int aAxis, int bAxis) {
        return tensorDot(src2, new int[]{aAxis}, new int[]{bAxis});
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     * @param src2 Tensor to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     * {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor.
     */
    T tensorDot(T src2, int[] aAxes, int[] bAxes);


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     * @param src2 Tensor to compute dot product with this tensor.
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     * along the second-to-last axis.
     */
    T tensorDot(T src2);


    /**
     * Computes the transpose of a tensor. Same as {@link #T(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link TensorOperationsMixin#transpose() transpose()} and
     * {@link TensorOperationsMixin#T() T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor.
     */
    default T transpose(int axis1, int axis2){
        return T(axis1, axis2);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link TensorOperationsMixin#transpose()} and
     * {@link TensorOperationsMixin#T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor.
     */
    T T(int axis1, int axis2);


    /**
     * Computes the transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation. Same as {@link #T(int[])}.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    default T transpose(int... axes){
        return T(axes);
    }


    /**
     * Computes the transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation. Same as {@link #transpose(int[])}.
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    T T(int... axes);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    T add(CooTensor B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    U add(Tensor B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    U sub(Tensor B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    CTensor add(CTensor B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    W add(CooCTensor B);


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    T sub(CooTensor B);


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    CTensor sub(CTensor B);


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    W sub(CooCTensor B);


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    T elemMult(Tensor B);


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    V elemMult(CooTensor B);


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    CTensor elemMult(CTensor B);


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    CooCTensor elemMult(CooCTensor B);


    /**
     * Computes the element-wise division between two tensors.
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    W elemDiv(CTensor B);


    /**
     * Computes the element-wise division between two tensors.
     * @param B Tensor to element-wise divide from this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    T elemDiv(Tensor B);


    /**
     * Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link #tensorDot(TensorBase, int) this.tensorDot(X, 2)} is the 'identity' tensor for the tensor dot product operation.
     * A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, 2).equals(this)}.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #tensorInv(int)
     */
    default U tensorInv(){
        return tensorInv(2);
    }


    /**
     * Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link #tensorDot(TensorBase, int) this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product operation.
     * A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #tensorInv()
     */
    U tensorInv(int numIndices);
}
