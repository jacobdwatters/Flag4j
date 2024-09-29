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


import org.flag4j.arrays.Shape;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.common.real.RealOperations;
import org.flag4j.operations.common.real.RealProperties;
import org.flag4j.operations.dense.real.RealDenseElemDiv;
import org.flag4j.operations.dense.real.RealDenseTensorDot;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

/**
 * <p>Base class for all real dense tensors which are backed by a primitive double array.</p>
 *
 * <p>The entries of DensePrimitiveDoubleTensorBase's are mutable but the tensor has a fixed shape.</p>
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}.
 * This type parameter is required because some operations (e.g. {@link #toCoo()}) result in a sparse tensor.
 */
public abstract class DensePrimitiveDoubleTensorBase <T extends DensePrimitiveDoubleTensorBase<T, U>,
        U extends SparseTensorMixin<T, U>>
        extends PrimitiveDoubleTensorBase<T, T>
        implements DenseTensorMixin<T, U> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected DensePrimitiveDoubleTensorBase(Shape shape, double[] entries) {
        super(shape, entries);
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
        return RealDenseTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
        return TransposeDispatcher.dispatchTensor(this, axis1, axis2);
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
        return TransposeDispatcher.dispatchTensor(this, axes);
    }


    /**
     * Computes the element-wise multiplication of two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void elemMultEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] *= b.entries[i];
    }


    /**
     * Computes the element-wise sum between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void addEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] += b.entries[i];
    }


    /**
     * Computes the element-wise difference between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void subEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] -= b.entries[i];
    }


    /**
     * Computes the element-wise division between two tensors and stores the result in this tensor.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @throws TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    @Override
    public void divEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] /= b.entries[i];
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor and {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    @Override
    public T div(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        return makeLikeTensor(shape, RealDenseElemDiv.dispatch(entries, shape, b.entries, b.shape));
    }


    /**
     * Rounds each entry of this tensor to the nearest whole number.
     *
     * @return A copy of this tensor with each entry rounded to the nearest whole number.
     * @see #round(int)
     * @see #roundToZero()
     * @see #roundToZero(double)
     */
    public T round() {
        return makeLikeTensor(this.shape, RealOperations.round(this.entries));
    }


    /**
     * Rounds each entry in this tensor to the nearest whole number.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     * @see #round()
     * @see #roundToZero()
     * @see #roundToZero(double)
     */
    @Override
    public T round(int precision) {
        return makeLikeTensor(this.shape, RealOperations.round(this.entries, precision));
    }


    /**
     * Rounds values in this tensor which are close to zero in absolute value to zero.
     * If the matrix is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within {@link Flag4jConstants#EPS_F64} of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this matrix with rounded values.
     * @see #roundToZero(double)
     * @see #round()
     * @see #round(int)
     */
    public T roundToZero() {
        this.abs();
        return makeLikeTensor(this.shape, RealOperations.roundToZero(this.entries, Flag4jConstants.EPS_F64));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently.
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     * @see #roundToZero()
     * @see #round()
     * @see #round(int)
     */
    public T roundToZero(double threshold) {
        return makeLikeTensor(this.shape, RealOperations.roundToZero(this.entries, threshold));
    }
    

    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     * @param tensor Tensor to compare this tensor to.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (1E-05 + 1E-08*|b|)}. Otherwise, returns false.
     * @see #allClose(DensePrimitiveDoubleTensorBase, double, double) 
     */
    public boolean allClose(T tensor) {
        return allClose(tensor, 1e-05, 1e-08);
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     * @param tensor Tensor to compare this tensor to.
     * @param absTol Absolute tolerance.
     * @param relTol Relative tolerance.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (atol + rtol*|b|)}. Otherwise, returns false.
     * @see #allClose(DensePrimitiveDoubleTensorBase) 
     */
    public boolean allClose(T tensor, double relTol, double absTol) {
        return shape.equals(tensor.shape) && RealProperties.allClose(entries, tensor.entries, relTol, absTol);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(Double value, int... indices) {
        ValidateParameters.ensureValidIndex(shape, indices);
        entries[shape.entriesIndex(indices)] = value;
        return (T) this;
    }


    /**
     * Checks if this tensor only contains positive values.
     * @return Returns {@code true} if this tensor only contains positive values. Otherwise, returns {@code false}.
     */
    public boolean isPos() {
        return RealProperties.isPos(entries);
    }


    /**
     * Checks if this tensor only contains negative values.
     * @return Returns {@code true} if this tensor only contains negative values. Otherwise, returns {@code false}.
     */
    public boolean isNeg() {
        return RealProperties.isNeg(entries);
    }
}
