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

package org.flag4j.core_old.dense_base;

import org.flag4j.arrays.Shape;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core_old.RealTensorMixin;
import org.flag4j.operations_old.common.complex.ComplexOperations;
import org.flag4j.operations_old.common.real.AggregateReal;
import org.flag4j.operations_old.common.real.RealOperations;
import org.flag4j.operations_old.common.real.RealProperties;
import org.flag4j.operations_old.dense.complex.ComplexDenseOperations;
import org.flag4j.operations_old.dense.real.*;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;

/**
 * The base class for all real dense tensors. This includes real dense matrices and vectors.
 * @param <T> Type of this tensor.
 * @param <W> Complex TensorOld type.
 */
public abstract class RealDenseTensorBase<T extends RealDenseTensorBase<T, W>, W extends ComplexDenseTensorBase<W, T>>
        extends DenseTensorBase<T, W, T, double[], Double>
        implements RealTensorMixin<T, W> {

    /**
     * Creates a real dense tensor with specified entries and shape.
     *
     * @param shape   Shape of this tensor.
     * @param entries Entries of this tensor. The number of entries must match the product of
     *                all {@code shape} dimensions.
     * @throws IllegalArgumentException If the number of entries does not equal the product of dimensions in the
     * {@code shape}.
     */
    protected RealDenseTensorBase(Shape shape, double[] entries) {
        super(shape, entries);
        ParameterChecks.ensureEquals(shape.totalEntries().intValueExact(), entries.length);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    protected abstract W makeComplexTensor(Shape shape, double[] entries);


    /**
     * Factory to create a complex tensor with the specified shape and size.
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    protected abstract W makeComplexTensor(Shape shape, CNumber[] entries);


    @Override
    public double min() {
        return AggregateReal.min(entries);
    }


    @Override
    public double max() {
        return AggregateReal.max(entries);
    }


    @Override
    public double minAbs() {
        return AggregateReal.minAbs(entries);
    }


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
        if(this.entries.length==0) {
            return new int[]{};
        } else {
            return shape.getIndices(AggregateDenseReal.argmin(entries));
        }
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        if(this.entries.length==0) {
            return new int[]{};
        } else {
            return shape.getIndices(AggregateDenseReal.argmax(entries));
        }
    }


    @Override
    public boolean isPos() {
        return RealProperties.isPos(entries);
    }


    @Override
    public boolean isNeg() {
        return RealProperties.isNeg(entries);
    }


    @Override
    public boolean isZeros() {
        return RealProperties.isZeros(entries);
    }


    @Override
    public boolean isOnes() {
        return RealDenseProperties.isOnes(entries);
    }


    @Override
    public T abs() {
        return makeTensor(shape, RealOperations.abs(entries));
    }


    @Override
    public T sqrt() {
        return makeTensor(shape, RealOperations.sqrt(entries));
    }


    @Override
    public T add(T B) {
        return makeTensor(shape, RealDenseOperations.add(entries, shape, B.entries, B.shape));
    }


    @Override
    public T sub(T B) {
        return makeTensor(shape, RealDenseOperations.sub(entries, shape, B.entries, B.shape));
    }


    @Override
    public T add(double b) {
        return makeTensor(shape, RealDenseOperations.add(entries, b));
    }


    @Override
    public T sub(double b) {
        return makeTensor(shape, RealDenseOperations.sub(entries, b));
    }


    @Override
    public T mult(double b) {
        return makeTensor(shape, RealOperations.scalMult(entries, b));
    }


    @Override
    public T div(double b) {
        return makeTensor(shape, RealDenseOperations.scalDiv(entries, b));
    }


    @Override
    public T elemMult(T B) {
        return makeTensor(shape, RealDenseElemMult.dispatch(entries, shape, B.entries, B.shape));
    }


    @Override
    public T elemDiv(T B) {
        return makeTensor(shape, RealDenseElemDiv.dispatch(entries, shape, B.entries, B.shape));
    }


    @Override
    public void addEq(T B) {
        RealDenseOperations.addEq(entries, shape, B.entries, B.shape);
    }


    @Override
    public void addEq(Double b) {
        RealDenseOperations.addEq(entries, b);
    }


    @Override
    public void subEq(T B) {
        RealDenseOperations.subEq(entries, shape, B.entries, B.shape);
    }


    @Override
    public void subEq(Double b) {
        RealDenseOperations.subEq(entries, b);
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        return AggregateReal.sum(entries);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException If the number of indices provided does not match the rank of this tensor.
     * @throws IllegalArgumentException If any of the indices are outside the tensor for that respective axis.
     */
    @Override
    public T set(double value, int... indices) {
        ParameterChecks.ensureArrayLengthsEq(indices.length, shape.getRank());
        RealDenseSetOperations.set(entries, shape, value, indices);
        return getSelf();
    }


    /**
     * Copies and reshapes tensor if possible. The total number of entries in this tensor must match the total number of entries
     * in the reshaped tensor.
     *
     * @param shape Shape of the new tensor.
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public T reshape(Shape shape) {
        ParameterChecks.ensureBroadcastable(this.shape, shape);
        return makeTensor(shape, this.entries.clone());
    }


    /**
     * Copies and reshapes tensor if possible. The total number of entries in this tensor must match the total number of entries
     * in the reshaped tensor.
     *
     * @param shape Shape of the new tensor.
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public T reshape(int... shape) {
        return reshape(new Shape(shape));
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public T recip() {
        return makeTensor(
                shape,
                RealDenseOperations.recip(entries)
        );
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.ensureArrayLengthsEq(indices.length, shape.getRank());
        return entries[shape.entriesIndex(indices)];
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeTensor(shape, entries.clone());
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public W toComplex() {
        return makeComplexTensor(shape, entries);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public W add(CNumber a) {
        return makeComplexTensor(
                shape,
                ComplexDenseOperations.add(entries, a)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public W sub(CNumber a) {
        return makeComplexTensor(
                shape,
                ComplexDenseOperations.sub(entries, a)
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public W mult(CNumber factor) {
        return makeComplexTensor(
                shape,
                ComplexOperations.scalMult(entries, factor)
        );
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public W div(CNumber divisor) {
        return makeComplexTensor(
                shape,
                RealComplexDenseOperations.scalDiv(entries, divisor)
        );
    }


    /**
     * Creates a hashcode for this matrix. Note, method adds {@link Arrays#hashCode(double[])} applied on the
     * underlying data array and the underlying shape array.
     * @return The hashcode for this matrix.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(entries)+Arrays.hashCode(shape.getDims());
    }


    @Override
    public T round() {
        return makeTensor(this.shape, RealOperations.round(this.entries));
    }


    @Override
    public T round(int precision) {
        return makeTensor(this.shape, RealOperations.round(this.entries, precision));
    }


    @Override
    public T roundToZero() {
        this.abs();
        return makeTensor(this.shape, RealOperations.roundToZero(this.entries, DEFAULT_ROUND_TO_ZERO_THRESHOLD));
    }


    @Override
    public T roundToZero(double threshold) {
        return makeTensor(this.shape, RealOperations.roundToZero(this.entries, threshold));
    }


    @Override
    public boolean allClose(T tensor, double relTol, double absTol) {
        return shape.equals(tensor.shape) && RealProperties.allClose(entries, tensor.entries, relTol, absTol);
    }
}


