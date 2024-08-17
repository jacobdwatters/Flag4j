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

package org.flag4j.core.dense_base;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.ComplexTensorMixin;
import org.flag4j.core.Shape;
import org.flag4j.operations_old.common.complex.AggregateComplex;
import org.flag4j.operations_old.common.complex.ComplexOperations;
import org.flag4j.operations_old.common.complex.ComplexProperties;
import org.flag4j.operations_old.dense.complex.*;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;

/**
 * The base class for all complex dense tensors. This includes complex dense matrices and vectors.
 * @param <T> Type of this tensor.
 * @param <Y> Real TensorOld type.
 */
public abstract class ComplexDenseTensorBase<T extends ComplexDenseTensorBase<T, Y>, Y extends RealDenseTensorBase<Y, T>>
        extends DenseTensorBase<T, T, Y, CNumber[], CNumber>
        implements ComplexTensorMixin<T, Y> {


    /**
     * Creates a complex dense tensor with specified entries and shape.
     *
     * @param shape   Shape of this tensor.
     * @param entries Entries of this tensor. The number of entries must match the product of
     *                all {@code shape} dimensions.
     * @throws IllegalArgumentException If the number of entries does not equal the product of dimensions in the
     * {@code shape}.
     */
    protected ComplexDenseTensorBase(Shape shape, CNumber[] entries) {
        super(shape, entries);
        ParameterChecks.assertEquals(shape.totalEntries().intValueExact(), entries.length);
    }


    /**
     * Factory to create a real tensor with the specified shape and size.
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    protected abstract Y makeRealTensor(Shape shape, double[] entries);


    @Override
    public double min() {
        return minAbs();
    }


    @Override
    public double max() {
        return maxAbs();
    }


    @Override
    public double minAbs() {
        return AggregateComplex.minAbs(entries);
    }


    @Override
    public double maxAbs() {
        return AggregateComplex.maxAbs(entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return shape.getIndices(AggregateDenseComplex.argMin(entries));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return shape.getIndices(AggregateDenseComplex.argMax(entries));
    }


    @Override
    public boolean isZeros() {
        return ComplexProperties.isZeros(entries);
    }


    @Override
    public boolean isOnes() {
        return ComplexDenseProperties.isOnes(entries);
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


    /**
     * Checks if this tensor has only real valued entries.
     *
     * @return True if this tensor contains <b>NO</b> complex entries. Otherwise, returns false.
     */
    @Override
    public boolean isReal() {
        return ComplexProperties.isReal(this.entries);
    }


    /**
     * Checks if this tensor contains at least one complex entry.
     *
     * @return True if this tensor contains at least one complex entry. Otherwise, returns false.
     */
    @Override
    public boolean isComplex() {
        return ComplexProperties.isComplex(this.entries);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public T set(double value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(indices.length, shape.getRank());
        ComplexDenseSetOperations.set(entries, shape, value, indices);
        return getSelf();
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    @Override
    public T set(CNumber value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(indices.length, shape.getRank());
        ComplexDenseSetOperations.set(entries, shape, value, indices);

        return getSelf();
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public CNumber get(int... indices) {
        ParameterChecks.assertArrayLengthsEq(indices.length, shape.getRank());
        return entries[shape.entriesIndex(indices)];
    }


    /**
     * Converts a complex tensor to a real matrix. The imaginary component of any complex value will be ignored.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     * @see #toRealSafe()
     */
    @Override
    public Y toReal() {
        return makeRealTensor(this.shape, ComplexOperations.toReal(this.entries));
    }


    /**
     * Converts a complex tensor to a real matrix safely. That is, first checks if the tensor only contains real values
     * and then converts to a real tensor. However, if non-real value exist, then an error is thrown.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     * @see #toReal()
     * @throws RuntimeException If this tensor contains at least one non-real value.
     */
    @Override
    public Y toRealSafe() {
        if(!this.isReal()) {
            throw new RuntimeException("Could not safely convert from complex to real as non-real " +
                    "values exist in tensor.");
        }

        return toReal();
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeTensor(this.shape, Arrays.copyOf(entries, entries.length));
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public T recip() {
        return makeTensor(this.shape, ComplexDenseOperations.recip(this.entries));
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public CNumber sum() {
        return AggregateComplex.sum(this.entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public T sqrt() {
        return makeTensor(this.shape, ComplexOperations.sqrt(this.entries));
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public Y abs() {
        return makeRealTensor(this.shape, ComplexOperations.abs(this.entries));
    }


    /**
     * Computes the complex conjugate of a tensor.
     *
     * @return The complex conjugate of this tensor.
     */
    @Override
    public T conj() {
        return makeTensor(this.shape, ComplexOperations.conj(this.entries));
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public T add(T B) {
        return makeTensor(
                this.shape,
                ComplexDenseOperations.add(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public T add(double a) {
        return makeTensor(this.shape, RealComplexDenseOperations.add(this.entries, a));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public T add(CNumber a) {
        return makeTensor(this.shape, ComplexDenseOperations.add(this.entries, a));
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public T sub(T B) {
        return makeTensor(
                this.shape,
                ComplexDenseOperations.sub(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public T sub(double a) {
        return makeTensor(this.shape,
                RealComplexDenseOperations.sub(this.entries, a)
        );
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public T sub(CNumber a) {
        return makeTensor(this.shape,
                ComplexDenseOperations.sub(this.entries, a)
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public T mult(double factor) {
        return makeTensor(this.shape,
                ComplexOperations.scalMult(this.entries, factor)
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public T mult(CNumber factor) {
        return makeTensor(this.shape,
                ComplexOperations.scalMult(this.entries, factor)
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
    public T div(double divisor) {
        return makeTensor(this.shape,
                RealComplexDenseOperations.scalDiv(this.entries, divisor)
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
    public T div(CNumber divisor) {
        return makeTensor(this.shape,
                ComplexDenseOperations.scalDiv(this.entries, divisor)
        );
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public T elemMult(T B) {
        return makeTensor(
                shape,
                ComplexDenseElemMult.dispatch(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B TensorOld to element-wise divide with this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public T elemDiv(T B) {
        return makeTensor(
                shape,
                ComplexDenseElemDiv.dispatch(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(T B) {
        ComplexDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(CNumber b) {
        ComplexDenseOperations.addEq(this.entries, b);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(Double b) {
        RealComplexDenseOperations.addEq(this.entries, b);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(T B) {
        ComplexDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(CNumber b) {
        ComplexDenseOperations.subEq(this.entries, b);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(Double b) {
        RealComplexDenseOperations.subEq(this.entries, b);
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
     * Copies and reshapes tensor if possible. The total number of entries in this tensor must match the total number of entries
     * in the reshaped tensor.
     *
     * @param shape Shape of the new tensor.
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public T reshape(Shape shape) {
        ParameterChecks.assertBroadcastable(this.shape, shape);
        return makeTensor(shape, this.entries.clone());
    }


    @Override
    public T round() {
        return makeTensor(this.shape, ComplexOperations.round(this.entries));
    }


    @Override
    public T round(int precision) {
        return makeTensor(this.shape, ComplexOperations.round(this.entries, precision));
    }


    @Override
    public T roundToZero() {
        this.abs();
        return makeTensor(this.shape, ComplexOperations.roundToZero(this.entries, DEFAULT_ROUND_TO_ZERO_THRESHOLD));
    }


    @Override
    public T roundToZero(double threshold) {
        return makeTensor(this.shape, ComplexOperations.roundToZero(this.entries, threshold));
    }


    @Override
    public boolean allClose(T tensor, double relTol, double absTol) {
        boolean close = shape.equals(tensor.shape);

        if(close) {
            for(int i=0; i<entries.length; i++) {
                double tol = absTol + relTol*tensor.entries[i].abs();

                if(entries[i].sub(tensor.entries[i]).abs() > tol) {
                    close = false;
                    break;
                }
            }
        }

        return close;
    }
}
