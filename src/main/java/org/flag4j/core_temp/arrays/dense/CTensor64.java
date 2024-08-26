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


import org.flag4j.core.Shape;
import org.flag4j.core_temp.FieldTensorBase;
import org.flag4j.core_temp.TensorPrimitiveOpsMixin;
import org.flag4j.core_temp.structures.fields.Complex128;
import org.flag4j.core_temp.structures.fields.Complex64;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.dense.field_ops.DenseFieldTensorDot;

import java.util.Arrays;

// TODO: Needs to implement DenseTensorMixin (Should probably have a DenseFieldTensorBase abstract class as well).
/**
 * Complex dense tensor backed by an array of {@link org.flag4j.core_temp.structures.fields.Complex64}'s.
 */
public class CTensor64 extends FieldTensorBase<CTensor64, CTensor64, Complex64> implements TensorPrimitiveOpsMixin<CTensor64> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected CTensor64(Shape shape, Complex64[] entries) {
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
    public CTensor64 tensorDot(CTensor64 src2, int[] aAxes, int[] bAxes) {
        return DenseFieldTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
    public CTensor64 tensorDot(CTensor64 src2) {
        return DenseFieldTensorDot.tensorDot(this, src2);
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
    public CTensor64 H(int axis1, int axis2) {
        return TransposeDispatcher.dispatchTensorHermitian(this, axis1, axis2);
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
    public CTensor64 H(int... axes) {
        return TransposeDispatcher.dispatchTensorHermitian(this, axes);
    }


    /**
     * Creates a zero tensor with the specified shape.
     *
     * @param shape Shape of this tensor.
     */
    public CTensor64(Shape shape) {
        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, Complex64.ZERO);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor64(Shape shape, Complex64 fillValue) {
        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor64(Shape shape, double fillValue) {
        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with. Must be a string representation of a complex number.
     */
    public CTensor64(Shape shape, String fillValue) {
        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, new Complex64(fillValue));
    }


    /**
     * <p>Creates a tensor with the specified entries and shape.</p>
     * <p>String array must contain properly formatted string representation of complex numbers.</p>
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @throws org.flag4j.util.exceptions.ComplexNumberParseingException If any entry in {@code entries} is not a properly formatted
     * string representation of complex number.
     */
    public CTensor64(Shape shape, String[] entries) {
        super(shape, new Complex64[entries.length]);

        // Parse string values.
        for(int i=0, size=entries.length; i<size; i++)
            super.entries[i] = new Complex64(entries[i]);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public CTensor64(Shape shape, float[] entries) {
        super(shape, new Complex64[entries.length]);

        // Wrap values as complex values.
        for(int i=0, size=entries.length; i<size; i++)
            super.entries[i] = new Complex64(entries[i]);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entires of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CTensor64 makeLikeTensor(Shape shape, Complex64[] entries) {
        return new CTensor64(shape, entries);
    }


    /**
     * Adds a scalar value to each element of this tensor.
     *
     * @param b Value to add to each entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public CTensor64 add(double b) {
        Complex64[] sum = new Complex64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            sum[i] = entries[i].add(b);

        return new CTensor64(shape, sum);
    }


    /**
     * Subtracts a scalar value from each element of this tensor.
     *
     * @param b Value to subtract from each entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public CTensor64 sub(double b) {
        Complex64[] diff = new Complex64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            diff[i] = entries[i].sub(b);

        return new CTensor64(shape, diff);
    }


    /**
     * Computes the sclar multiplication between this tensor and the specified scalar {@code factor}.
     *
     * @param factor Scalar factor to apply to this tensor.
     *
     * @return The sclar product of this tensor and {@code factor}.
     */
    @Override
    public CTensor64 mult(double factor) {
        Complex64[] product = new Complex64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            product[i] = entries[i].mult(factor);

        return new CTensor64(shape, product);
    }


    /**
     * Computes the scalar division of this tensor and the specified scalar {@code factor}.
     *
     * @param divisor The scalar value to divide this tensor by.
     *
     * @return The result of dividing this tensor by the specified scalar.
     */
    @Override
    public CTensor64 div(double divisor) {
        Complex64[] quotient = new Complex64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            quotient[i] = entries[i].div(divisor);

        return new CTensor64(shape, quotient);
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
    public CTensor64 T(int axis1, int axis2) {
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
    public CTensor64 T(int... axes) {
        return TransposeDispatcher.dispatchTensor(this, axes);
    }
}
