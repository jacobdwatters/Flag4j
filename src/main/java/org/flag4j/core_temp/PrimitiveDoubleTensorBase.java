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
import org.flag4j.operations.common.real.AggregateReal;
import org.flag4j.operations.common.real.RealOperations;
import org.flag4j.operations.common.real.RealProperties;
import org.flag4j.operations.dense.real.AggregateDenseReal;
import org.flag4j.operations.dense.real.RealDenseElemMult;
import org.flag4j.operations.dense.real.RealDenseOperations;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;


/**
 * <p>Base class for all real tensors which are backed by a primitive double array.</p>
 *
 * <p>The entries of PrimitiveDoubleTensorBase's are mutable but the tensor has a fixed shape.</p>
 *
 * @param <T> Type of this tensor.
 * @param <U> Type of a dense tensor equivalent to {@code T}. If {@code T} is dense, then this should be the same type as {@code T}.
 * This type parameter is required becase some operations (e.g. {@link #tensorDot(TensorOverSemiRing)}) between two sparse
 * tensors result in a dense tensor.
 */
public abstract class PrimitiveDoubleTensorBase<T extends PrimitiveDoubleTensorBase<T, U>,
        U extends PrimitiveDoubleTensorBase<U, U>>
        extends TensorOverField<T, U, double[], Double>
        implements TensorPrimitiveOpsMixin<T> {


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected PrimitiveDoubleTensorBase(Shape shape, double[] entries) {
        super(shape, entries);
        ParameterChecks.ensureEquals(shape.totalEntries().intValueExact(), entries.length);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.ensureValidIndex(shape, indices);
        return entries[shape.entriesIndex(indices)];
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return makeLikeTensor(new Shape(entries.length), entries.clone());
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(dims);

        return makeLikeTensor(flatShape, entries.clone());
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
        ParameterChecks.ensureBroadcastable(this.shape, shape);
        return makeLikeTensor(shape, this.entries.clone());
    }


    /**
     * Subtracts a sclar value from each entry of this tensor.
     *
     * @param b Scalar value in differencce.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(Double b) {
        return sub(b.doubleValue());
    }


    /**
     * Subtracts a sclar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in differencce.
     */
    @Override
    public void subEq(Double b) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] -= b;
    }


    /**
     * Adds a sclar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(Double b) {
        return add(b.doubleValue());
    }


    /**
     * Adds a sclar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        double bUnboxed = b;

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] += bUnboxed;
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        return makeLikeTensor(shape, RealDenseOperations.add(entries, shape, b.entries, b.shape));
    }


    /**
     * Multiplies a sclar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public T mult(Double b) {
        return mult(b.doubleValue());
    }


    /**
     * Multiplies a sclar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(Double b) {
        double bUnboxed = b;

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] *= bUnboxed;
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        return makeLikeTensor(shape, RealDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public T abs() {
        return makeLikeTensor(shape, RealOperations.abs(entries));
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
        return makeLikeTensor(shape, RealDenseElemMult.dispatch(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. If {@code axes=N}, then
     * the product sums will be computed along the last {@code N} dimensions of this tensor and the first {@code N} dimensions of
     * the {@code src2} tensor.
     *
     * @param src2 TensorOld to contract with this tensor.
     * @param axes Axes specifying the number of axes to compute the tensor dot product over. If {@code axes=N}, then
     * the product sums will be computed along the last {@code N} dimensions of this tensor and the first {@code N}
     * dimensions of.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     *                                  and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    @Override
    public U tensorDot(T src2, int axes) {
        return super.tensorDot(src2, axes);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. That is,
     * computes the sum of products between the two tensors along the specified axes.
     *
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxis Axis along which to compute products for this tensor.
     * @param bAxis Axis along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     *                                  and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    @Override
    public U tensorDot(T src2, int aAxis, int bAxis) {
        return super.tensorDot(src2, aAxis, bAxis);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specifieed by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}. This will be a tensor of rank
     * {@code this.getRank() - 2} with the same shape as this tensor but with {@code axis1} and {@code axis2} removed.
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     * (i.e. the axes are equal or the tesnor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        ParameterChecks.ensureNotEquals(axis1, axis2);
        ParameterChecks.ensureValidIndices(getRank(), axis1, axis2);
        ParameterChecks.ensureEquals(shape.get(axis1), shape.get(axis2));

        int[] strides = shape.getStrides();
        int rank = strides.length;
        int[] newDims = new int[rank - 2];

        int idx = 0;

        // Compute shape for resulting tensor.
        for(int i=0; i<rank; i++) {
            if(i != axis1 && i != axis2) newDims[idx++] = shape.get(i);
        }

        Shape destShape = new Shape(newDims);
        double[] destEntries = new double[destShape.totalEntries().intValueExact()];

        // Calculate the offset increment for the diagonal.
        int traceLength = shape.get(axis1);
        int diagonalStride = strides[axis1] + strides[axis2];

        int[] destIndices = new int[rank - 2];
        for(int i=0; i<destEntries.length; i++) {
            destIndices = destShape.getIndices(i);

            int baseOffset = 0;
            idx = 0;

            // Compute offset for mapping destination indices to indices in this tensor.
            for(int j=0; j<rank; j++) {
                if(j != axis1 && j != axis2) {
                    baseOffset += destIndices[idx++]*strides[j];
                }
            }

            // Sum over diagonal elements of the 2D sub-array.
            double sum = 0;
            int offset = baseOffset;
            for(int diag=0; diag<traceLength; diag++) {
                sum += entries[offset];
                offset += diagonalStride;
            }

            destEntries[i] = sum;
        }

        return makeLikeTensor(destShape, destEntries);
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return RealProperties.isZeros(entries);
    }


    /**
     * Computes the sum of all values in this tensor.
     *
     * @return The sum of all values in this tensor.
     */
    @Override
    public Double sum() {
        return AggregateReal.sum(entries);
    }


    /**
     * Computes the product of all values in this tensor.
     *
     * @return The product of all values in this tensor.
     */
    @Override
    public Double prod() {
        return AggregateReal.prod(entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public T sqrt() {
        return makeLikeTensor(shape, RealOperations.sqrt(entries));
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor..
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public T T() {
        return T(0, shape.getRank()-1);
    }


    /**
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    public T conj() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public T H() {
        return T();
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        return T(axis1, axis2);
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


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     */
    @Override
    public T recip() {
        return makeLikeTensor(shape, RealDenseOperations.recip(entries));
    }


    /**
     * Divides each entry of this tensor by a scalar field element.
     *
     * @param b Scalar field value in quotient.
     *
     * @return The quotient of this tensor with {@code b}.
     */
    @Override
    public T div(Double b) {
        return div(b.doubleValue());
    }


    /**
     * Divides each entry of this tensor by a scalar field element and stores the result in this tensor.
     *
     * @param b Scalar field value in quotient.
     */
    @Override
    public void divEq(Double b) {
        double bUnboxed = b;

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] /= bUnboxed;
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, entries.clone());
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double min() {
        return AggregateReal.min(entries);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double max() {
        return AggregateReal.max(entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return AggregateReal.minAbs(entries);
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return AggregateReal.maxAbs(entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        if(this.entries.length==0) return new int[]{};
        else return shape.getIndices(AggregateDenseReal.argmin(entries));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        if(this.entries.length==0) return new int[]{};
        else return shape.getIndices(AggregateDenseReal.argmax(entries));
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        if(this.entries.length==0) return new int[]{};
        return shape.getIndices(AggregateDenseReal.argminAbs(entries));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return shape.getIndices(AggregateDenseReal.argmaxAbs(entries));
    }


    /**
     * Adds a scalar value to each element of this tensor.
     *
     * @param b Value to add to each entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public T add(double b) {
        return makeLikeTensor(shape, RealDenseOperations.add(entries, b));
    }


    /**
     * Subtracts a scalar value from each element of this tensor.
     *
     * @param b Value to subtract from each entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public T sub(double b) {
        return makeLikeTensor(shape, RealDenseOperations.sub(entries, b));
    }


    /**
     * Computes the sclar multiplication between this tensor and the specified scalar {@code factor}.
     *
     * @param factor Scalar factor to apply to this tensor.
     *
     * @return The sclar product of this tensor and {@code factor}.
     */
    @Override
    public T mult(double factor) {
        return makeLikeTensor(shape, RealOperations.scalMult(entries, factor));
    }


    /**
     * Computes the scalar division of this tensor and the specified scalar {@code divisor}.
     *
     * @param divisor The scalar value to divide this tensor by.
     *
     * @return The result of dividing this tensor by the specified scalar.
     */
    @Override
    public T div(double divisor) {
        return makeLikeTensor(shape, RealOperations.scalDiv(entries, divisor));
    }


    /**
     * Rounds all entries of this tensor to the nearest whole number.
     * @return A copy of this tensor with all entries rounded to the nearest whole number.
     * @see #round(int)
     */
    public T round() {
        return makeLikeTensor(shape, RealOperations.round(entries));
    }


    /**
     * Rounds all entries of this tensor to the number of decimal specified by {@code precision}.
     * @param precision The number of decimal places to round the tensor to.
     * @return A copy of this tensor with all entries rounded to the number of decimal specified by {@code precision}.
     * @see #round()
     */
    public T round(int precision) {
        return makeLikeTensor(shape, RealOperations.round(entries, precision));
    }
}
