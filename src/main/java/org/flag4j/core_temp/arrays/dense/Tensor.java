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

package org.flag4j.core_temp.arrays.dense;


import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.core.Shape;
import org.flag4j.core_temp.TensorOverField;
import org.flag4j.core_temp.TensorOverSemiRing;
import org.flag4j.core_temp.TensorPrimitiveOpsMixin;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.common.real.AggregateReal;
import org.flag4j.operations.common.real.RealOperations;
import org.flag4j.operations.dense.real.*;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;


/**
 * A real dense tensor backed by a primative double array.
 */
public class Tensor extends TensorOverField<Tensor, double[], Double> implements TensorPrimitiveOpsMixin<Tensor> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public Tensor(Shape shape, double[] entries) {
        super(shape, entries);
        ParameterChecks.assertEquals(shape.totalEntries().intValueExact(), entries.length);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws IllegalArgumentException       If {@code indices} is not of length 2.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.assertValidIndex(shape, indices);
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
    public Tensor flatten() {
        return new Tensor(new Shape(entries.length), entries.clone());
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
    public Tensor flatten(int axis) {
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(true, dims);

        return new Tensor(flatShape, entries.clone());
    }


    /**
     * Subtracts a sclar value from each entry of this tensor.
     *
     * @param b Scalar value in differencce.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public Tensor sub(Double b) {
        return sub(b.doubleValue());
    }


    /**
     * Adds a sclar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public Tensor add(Double b) {
        return add(b.doubleValue());
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
    public Tensor add(Tensor b) {
        return new Tensor(shape, RealDenseOperations.add(entries, shape, b.entries, b.shape));
    }


    /**
     * Multiplies a sclar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public Tensor mult(Double b) {
        return mult(b.doubleValue());
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
    public Tensor sub(Tensor b) {
        return new Tensor(shape, RealDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Tensor abs() {
        return new Tensor(shape, RealOperations.abs(entries));
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
    public Tensor elemMult(Tensor b) {
        return new Tensor(shape, RealDenseElemMult.dispatch(entries, shape, b.entries, b.shape));
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
    public Tensor tensorDot(Tensor src2, int[] aAxes, int[] bAxes) {
        return RealDenseTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     *
     * @param src2 TensorOld to compute dot product with this tensor.
     *
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     *
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     *                                  along the second-to-last axis.
     */
    @Override
    public Tensor tensorDot(Tensor src2) {
        return RealDenseTensorDot.tensorDot(this, src2);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specifieed by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1
     * @param axis2
     *
     * @return The generalized trace of this tensor along @link axis1} and {@code axis2}.
     */
    @Override
    public Tensor tensorTr(int axis1, int axis2) {
        return null;
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public Tensor sqrt() {
        return new Tensor(shape, RealOperations.sqrt(entries));
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
    public Tensor T() {
        return TransposeDispatcher.dispatchTensor(this, 0, shape.getRank()-1);
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public Tensor T(int axis1, int axis2) {
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
    public Tensor T(int... axes) {
        return TransposeDispatcher.dispatchTensor(this, axes);
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     */
    @Override
    public Tensor recip() {
        return new Tensor(shape, RealDenseOperations.recip(entries));
    }


    /**
     * Divides each entry of this tensor by s scalar field element.
     *
     * @param b Scalar field value in quotient.
     *
     * @return The quotient of this tensor with {@code b}.
     */
    @Override
    public Tensor div(Double b) {
        return div(b.doubleValue());
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public Tensor copy() {
        return new Tensor(shape, entries.clone());
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
    public int[] argMin() {
        if(this.entries.length==0) return new int[]{};
        else return shape.getIndices(AggregateDenseReal.argMin(entries));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        if(this.entries.length==0) return new int[]{};
        else return shape.getIndices(AggregateDenseReal.argMax(entries));
    }


    /**
     * Finds the indices of the minimum absollte value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMinAbs() {
        // TODO: Implementation.
        return new int[0];
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMaxAbs() {
        // TODO: Implementation.
        return new int[0];
    }


    /**
     * Adds a scalar value to each element of this tensor.
     *
     * @param b Value to add to each entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public Tensor add(double b) {
        return new Tensor(shape, RealDenseOperations.add(entries, b));
    }


    /**
     * Subtracts a scalar value from each element of this tensor.
     *
     * @param b Value to subtract from each entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public Tensor sub(double b) {
        return new Tensor(shape, RealDenseOperations.sub(entries, b));
    }


    /**
     * Computes the sclar multiplication between this tensor and the specified scalar {@code factor}.
     *
     * @param factor Scalar factor to apply to this tensor.
     *
     * @return The sclar product of this tensor and {@code factor}.
     */
    @Override
    public Tensor mult(double factor) {
        return new Tensor(shape, RealOperations.scalMult(entries, factor));
    }


    /**
     * Computes the scalar division of this tensor and the specified scalar {@code factor}.
     *
     * @param divisor The scalar value to divide this tensor by.
     *
     * @return The result of dividing this tensor by the specified scalar.
     */
    @Override
    public Tensor div(double divisor) {
        return new Tensor(shape, RealOperations.scalDiv(entries, divisor));
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.inv(numIndices)} such that
     * {@link #tensorDot(TensorOverSemiRing, int) this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product
     * operation.</p>
     *
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.</p>
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #inv()
     */
    public Tensor inv(int numIndices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.inv()} such that
     * {@link #tensorDot(Tensor) this.tensorDot(X)} is the 'identity' tensor for the tensor dot product
     * operation.</p>
     *
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I).equals(this)}.</p>
     * 
     * <p>Equivalent to {@link #inv(int) inv(2)}.</p>
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #inv(int)
     */
    public Tensor inv() {
        return inv(2);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link Tensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        TensorOld src2 = (TensorOld) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }
}
