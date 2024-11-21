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

package org.flag4j.arrays.backend_new.primitive;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.linalg.operations.TransposeDispatcher;
import org.flag4j.linalg.operations.common.real.RealProperties;
import org.flag4j.linalg.operations.dense.real.RealDenseElemDiv;
import org.flag4j.linalg.operations.dense.real.RealDenseElemMult;
import org.flag4j.linalg.operations.dense.real.RealDenseOperations;
import org.flag4j.linalg.operations.dense.real.RealDenseTensorDot;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringOperations;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

/**
 * This is the base class of all real primitive double tensors. The methods implemented in this class are agnostic to weather the
 * tensor is dense or sparse.
 * @param <T> The type of the tensor.
 */
public abstract class AbstractDenseDoubleTensor<T extends AbstractDoubleTensor<T>>
        extends AbstractDoubleTensor<T> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractDenseDoubleTensor(Shape shape, double[] entries) {
        super(shape, entries);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    @Override
    public Double get(int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        return entries[shape.getFlatIndex(indices)];
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return A reference to this tensor.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(Double value, int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        entries[shape.getFlatIndex(indices)] = value;
        return (T) this;
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public T T(int axis1, int axis2) {
        return TransposeDispatcher.dispatchTensor((T) this, axis1, axis2);
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public T T(int... axes) {
        return makeLikeTensor(shape.permuteAxes(axes),
                TransposeDispatcher.dispatchTensor(entries, shape, axes));
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        return makeLikeTensor(shape, RealDenseOperations.sub(entries, b.entries, null));
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void subEq(T b) {
        RealDenseOperations.sub(entries, b.entries, entries);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        return makeLikeTensor(shape, RealDenseOperations.add(entries,b.entries, null));
    }

    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     */
    public void addEq(T b) {
        RealDenseOperations.add(entries,b.entries, entries);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T elemMult(T b) {
        return makeLikeTensor(shape, RealDenseElemMult.elemMult(entries,b.entries, null));
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public void elemMultEq(T b) {
        RealDenseElemMult.elemMult(entries,b.entries, entries);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public T tensorDot(T src2, int[] aAxes, int[] bAxes) {
        RealDenseTensorDot problem = new RealDenseTensorDot(
                shape, entries, src2.shape, src2.entries, aAxes, bAxes);
        return makeLikeTensor(problem.getOutputShape(), problem.compute());
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public Tensor tensorTr(int axis1, int axis2) {
        Shape destShape = DenseSemiringOperations.getTrShape(shape, axis1, axis2);
        double[] destEntries = new double[destShape.totalEntriesIntValueExact()];
        RealDenseOperations.tensorTr(shape, entries, axis1, axis2, destShape, destEntries);
        return new Tensor(destShape, destEntries);
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    @Override
    public T div(T b) {
        return makeLikeTensor(shape, RealDenseElemDiv.dispatch(entries, b.entries, null));
    }


    /**
     * Computes the element-wise quotient between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise quotient.
     */
    public void divEq(T b) {
        RealDenseElemDiv.dispatch(entries, b.entries, entries);
    }



    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return shape.getNdIndices(RealProperties.argminAbs(entries));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return shape.getNdIndices(RealProperties.argmaxAbs(entries));
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        return shape.getNdIndices(RealProperties.argmin(entries));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return shape.getNdIndices(RealProperties.argmax(entries));
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        ValidateParameters.ensureBroadcastable(shape, newShape);
        return makeLikeTensor(newShape, entries.clone());
    }
}
