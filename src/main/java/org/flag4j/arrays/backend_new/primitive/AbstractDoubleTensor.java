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
import org.flag4j.arrays.backend_new.AbstractTensor;
import org.flag4j.arrays.backend_new.field.TensorOverField;
import org.flag4j.linalg.operations.common.real.AggregateReal;
import org.flag4j.linalg.operations.common.real.RealOperations;
import org.flag4j.linalg.operations.common.real.RealProperties;
import org.flag4j.linalg.operations.dense.real.RealDenseOperations;
import org.flag4j.util.Flag4jConstants;

/**
 * This is the base class of all real primitive double tensors, matrices, or vectors. The methods implemented in this class are
 * agnostic to weather the tensor is dense or sparse.
 */
public abstract class AbstractDoubleTensor<T extends AbstractDoubleTensor<T>>
        extends AbstractTensor<T, double[], Double>
        implements TensorOverField<T, T, double[], Double> {

    // TODO: Adjust method JavaDocs to reflect that it may compute a solution only using non-zero
    //  values if the tensor is sparse.

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractDoubleTensor(Shape shape, double[] entries) {
        super(shape, entries);
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
     * Subtracts a scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(Double b) {
        return sub((double) b);
    }


    /**
     * Subtracts a scalar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Double b) {
        subEq((double) b);
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
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    public T conj() {
        return copy();
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
        return H(axes);
    }


    /**
     * Finds the minimum value in this tensor.
     *
     * @return The minimum value in this tensor.
     */
    @Override
    public Double min() {
        return RealProperties.min(entries);
    }


    /**
     * Finds the maximum value in this tensor.
     *
     * @return The maximum value in this tensor.
     */
    @Override
    public Double max() {
        return RealProperties.max(entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return RealProperties.minAbs(entries);
    }


    /**
     * Finds the maximum absolute value in this tensor.
     *
     * @return The maximum absolute value in this tensor.
     */
    @Override
    public double maxAbs() {
        return RealProperties.maxAbs(entries);
    }


    /**
     * Adds a scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the non-zero
     * entries of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(Double b) {
        return add((double) b);
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        addEq((double) b);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public T mult(Double b) {
        return mult((double) b);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(Double b) {
        multEq((double) b);
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
     * Checks if this tensor only contains ones. If this tensor is sparse, only the non-zero entries are considered.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return RealProperties.isOnes(entries);
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
        return AggregateReal.sum(entries);
    }


    /**
     * Adds a primitive scalar value to each entry of this tensor. If the tensor is sparse, the scalar will only be added to the
     * non-zero entries of the tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(double b) {
        return makeLikeTensor(shape, RealDenseOperations.add(entries, b, null));
    }


    /**
     * Adds a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(double b) {
        RealDenseOperations.add(entries, b, entries);
    }


    /**
     * Multiplies a primitive scalar value to each entry of this tensor.
     *
     * @param b Scalar value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public T mult(double b) {
        return makeLikeTensor(shape, RealOperations.scalMult(entries, b, null));
    }


    /**
     * Multiplies a primitive scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(double b) {
        RealOperations.scalMult(entries, b, entries);
    }


    /**
     * Subtracts a primitive scalar value from each entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(double b) {
        return makeLikeTensor(shape, RealDenseOperations.sub(entries, b, null));
    }


    /**
     * Subtracts a scalar primitive value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(double b) {
        RealDenseOperations.sub(entries, b, entries);
    }


    /**
     * Divides each element of this tensor by a scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(double)
     */
    @Override
    public T div(Double b) {
        return div((double) b);
    }


    /**
     * Divides each element of this tensor by a scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #div(double)
     */
    @Override
    public void divEq(Double b) {
        divEq((double) b);
    }


    /**
     * Divides each element of this tensor by a primitive scalar value.
     *
     * @param b Scalar value in quotient.
     *
     * @return The element-wise quotient of this tensor and the scalar {@code b}.
     *
     * @see #divEq(double)
     */
    @Override
    public T div(double b) {
        return makeLikeTensor(shape, RealOperations.scalDiv(entries, b, null));
    }


    /**
     * Divides each element of this tensor by a primitive scalar value and stores the result in this tensor.
     *
     * @param b Scalar value in quotient.
     *
     * @see #div(double)
     */
    @Override
    public void divEq(double b) {
        RealOperations.scalDiv(entries, b, entries);
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        return makeLikeTensor(shape, RealOperations.sqrt(entries));
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return The element-wise reciprocals of this tensor.
     */
    @Override
    public T recip() {
        return makeLikeTensor(shape, RealDenseOperations.recip(entries));
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values. Otherwise, returns {@code false}.
     *
     * @see #isInfinite()
     * @see #isNaN()
     */
    @Override
    public boolean isFinite() {
        return RealProperties.isFinite(entries);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value. Otherwise, returns {@code false}.
     *
     * @see #isFinite()
     * @see #isNaN()
     */
    @Override
    public boolean isInfinite() {
        return RealProperties.isInfinite(entries);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value. Otherwise, returns {@code false}.
     *
     * @see #isFinite()
     * @see #isInfinite()
     */
    @Override
    public boolean isNaN() {
        return RealProperties.isNaN(entries);
    }
}
