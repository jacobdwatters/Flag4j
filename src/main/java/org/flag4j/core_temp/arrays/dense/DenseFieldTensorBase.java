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
import org.flag4j.core_temp.arrays.sparse.SparseTensorMixin;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.core_temp.structures.fields.RealFloat64;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.dense.field_ops.DenseFieldEquals;
import org.flag4j.operations.dense.field_ops.DenseFieldTensorDot;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.TensorShapeException;


/**
 * <p>The base class for all dense {@link Field} tensors.</p>
 * <p>Then {@link #entries} of a DenseFieldTensor are mutable but the {@link #shape} is fixed.</p>
 *
 * @param <T> The type of this dense field tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #toCoo()}) may result in a sparse tensor.
 * @param <V> The type of the {@link Field field} which this tensor's entries belong to.
 */
public abstract class DenseFieldTensorBase<T extends DenseFieldTensorBase<T, U, V>,
        U extends SparseTensorMixin<T, U>, V extends Field<V>>
        extends FieldTensorBase<T, T, V>
        implements DenseTensorMixin<T, U> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected DenseFieldTensorBase(Shape shape, V[] entries) {
        super(shape, entries);
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public FieldTensor<RealFloat64> abs() {
        RealFloat64[] abs = new RealFloat64[entries.length];
        for(int i = 0, size=entries.length; i<size; ++i)
            abs[i] = new RealFloat64(entries[i].abs());

        return new FieldTensor<RealFloat64>(shape, abs);
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
    public T H(int... axes) {
        return TransposeDispatcher.dispatchTensorHermitian(this, axes);
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
        return DenseFieldTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
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
        ParameterChecks.ensurePermutation(axes);
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
        ParameterChecks.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].mult(b.entries[i]);
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
        ParameterChecks.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].add(b.entries[i]);
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
        ParameterChecks.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].sub(b.entries[i]);
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
        ParameterChecks.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].div(b.entries[i]);
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
        ParameterChecks.ensureEqualShape(this.shape, b.shape);
        Field<V>[] quotient = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            quotient[i] = entries[i].div(b.entries[i]);

        return makeLikeTensor(shape, (V[]) quotient);
    }


    /**
     * Checks if all entries of this matrix are close to the entries of {@code b}.
     *
     * @param b Matrix to compare this tensor to.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     *
     * @return True if {@code b} is the same shape as this matrix and all entries are 'close', i.e.
     * elements {@code x} and {@code y} at the same positions in the two matrices respectively satisfy
     * {@code |ax-y| <= (atol + rtol*|y|)}. Otherwise, returns false.
     *
     * @see #allClose(DenseTensorMixin)
     */
    @Override
    public boolean allClose(T b, double relTol, double absTol) {
        return DenseFieldEquals.allClose(entries, shape, b.entries, b.shape, relTol, absTol);
    }
}
