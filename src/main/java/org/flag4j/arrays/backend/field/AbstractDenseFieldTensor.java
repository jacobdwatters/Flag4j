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

package org.flag4j.arrays.backend.field;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.backend.SparseTensorData;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.field_ops.FieldProperties;
import org.flag4j.linalg.ops.dense.DenseSemiringTensorDot;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldElemDiv;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.dense.ring_ops.DenseRingTensorOps;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringConversions;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringElemMult;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringOperations;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;

/**
 * <p>The base class for all dense {@link Field} tensors.</p>
 * <p>The {@link #data} of an AbstractDenseFieldTensor are mutable but the {@link #shape} is fixed.</p>
 *
 * @param <T> The type of this dense field tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}. This type parameter is required because some ops (e.g.
 * {@link #toCoo()}) may result in a sparse tensor.
 * @param <V> The type of the {@link Field} which this tensor's data belong to.
 */
public abstract class AbstractDenseFieldTensor<T extends AbstractDenseFieldTensor<T, V>, V extends Field<V>>
        extends AbstractTensor<T, V[], V>
        implements FieldTensorMixin<T, T, V> {

    /**
     * The zero element for the field that this tensor's elements belong to.
     */
    private V zeroElement;

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseFieldTensor(Shape shape, V[] entries) {
        super(shape, entries);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        this.zeroElement = (entries.length > 0 && entries[0] != null) ? entries[0].getZero() : null;
    }


    /**
     * Sets the zero element for the field of this tensor.
     * @param zeroElement The zero element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the field.
     *
     * @see #getZeroElement()
     */
    public void setZeroElement(V zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }


    /**
     * Gets the zero element for the field of this tensor.
     * @return The zero element for the field of this tensor. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Field)} then {@code null} will be returned.
     *
     * @see #setZeroElement(Field)
     */
    public V getZeroElement() {
        return zeroElement;
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param rowIndices Non-zero row indices of the COO tensor.
     * @param colIndices Non-zero column indices of the COO tensor.
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    protected abstract AbstractTensor<?, V[], V> makeLikeCooTensor(
            Shape shape, V[] entries, int[][] indices);


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
    public V get(int... indices) {
        return data[shape.getFlatIndex(indices)];
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
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensor(data, shape, axis1, axis2, dest);
        return makeLikeTensor(shape.swapAxes(axis1, axis2), dest);
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
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensor(data, shape, axes, dest);
        return makeLikeTensor(shape.permuteAxes(axes), dest);
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, data.clone());
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
    public T set(V value, int... indices) {
        data[shape.getFlatIndex(indices)] = value;
        return (T) this;
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        return makeLikeTensor(new Shape(shape.totalEntriesIntValueExact()), data.clone());
    }


    /**
     * Flattens a tensor along the specified axis. Unlike {@link #flatten()}
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(dims);

        return makeLikeTensor(flatShape, data.clone());
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
        // No need to make explicit broadcastable check as the constructor will verify that the number of data in the shape matches
        // the number of data in the array.
        return makeLikeTensor(newShape, data.clone());
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
        V[] diff = makeEmptyDataArray(data.length);
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, diff);
        return makeLikeTensor(shape, diff);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void subEq(T b) {
        DenseRingTensorOps.sub(shape, data, b.shape, b.data, data);
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
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensorHermitian(shape, data, axis1, axis2, dest);
        return makeLikeTensor(shape.swapAxes(axis1, axis2), dest);
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
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchTensorHermitian(shape, data, axes, dest);
        return makeLikeTensor(shape.permuteAxes(axes), dest);
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
        V[] sum = makeEmptyDataArray(data.length);
        DenseSemiringOperations.add(data, shape, b.data, b.shape, sum);
        return makeLikeTensor(shape, sum);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     */
    public void addEq(T b) {
        DenseSemiringOperations.add(data, shape, b.data, b.shape, data);
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
        V[] prod = makeEmptyDataArray(data.length);
        DenseSemiringElemMult.dispatch(data, shape, b.data, b.shape, prod);
        return makeLikeTensor(shape, prod);
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
        DenseSemiringTensorDot<V> dot = new DenseSemiringTensorDot(shape, data, src2.shape, src2.data, aAxes, bAxes);
        V[] dest = makeEmptyDataArray(dot.getOutputSize());
        dot.compute(dest);
        return makeLikeTensor(dot.getOutputShape(), dest);
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
    public T tensorTr(int axis1, int axis2) {
        Shape destShape = DenseSemiringOperations.getTrShape(shape, axis1, axis2);
        V[] dest = makeEmptyDataArray(destShape.totalEntriesIntValueExact());
        return makeLikeTensor(destShape, dest);
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
        V[] dest = makeEmptyDataArray(data.length);
        DenseFieldElemDiv.dispatch(data, shape, b.data, b.shape, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.sqrt(data, dest);
        return makeLikeTensor(shape, dest);
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
        return FieldOps.isFinite(data);
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
        return FieldOps.isInfinite(data);
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
        return FieldOps.isInfinite(data);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public AbstractTensor<?, ?, V> toCoo() {
        return toCoo(0.9);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public AbstractTensor<?, ?, V> toCoo(double estimatedSparsity) {
        SparseTensorData<V> data = DenseSemiringConversions.toCooTensor(shape, this.data, estimatedSparsity);
        V[] cooEntries = data.data().toArray(makeEmptyDataArray(data.data().size()));

        // TODO: First check if this tensor is a vector then delegate to specialized toCooVector
        //  or toCooTensor methods.
        if(this instanceof VectorMixin<?,?,?,?>) {
            return makeLikeCooTensor(
                    data.shape(), cooEntries,
                    RealDenseTranspose.standardIntMatrix(data.indicesToArray()));
        } else {
            return makeLikeCooTensor(data.shape(), cooEntries, data.indicesToArray());
        }
    }


    /**
     * Checks if all data of this matrix are 'close' as defined below. Custom tolerances may be specified using
     * {@link #allClose(AbstractDenseFieldTensor, double, double)}.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same shape and all data are 'close' element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (1E-05 + 1E-08*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldTensor, double, double)
     */
    public boolean allClose(T b) {
        return sameShape(b) && FieldProperties.allClose(data, b.data);
    }


    /**
     * Checks if all data of this matrix are 'close' as defined below.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same length and all data are 'close' element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (absTol + relTol*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldTensor)
     */
    public boolean allClose(T b, double relTol, double absTol) {
        return sameShape(b) && FieldProperties.allClose(data, b.data, relTol, absTol);
    }
}
