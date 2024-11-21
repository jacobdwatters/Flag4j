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

package org.flag4j.arrays.backend_new.ring;

import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.AbstractTensor;
import org.flag4j.arrays.backend_new.SparseTensorData;
import org.flag4j.linalg.operations.TransposeDispatcher;
import org.flag4j.linalg.operations.common.ring_ops.CompareRing;
import org.flag4j.linalg.operations.dense.DenseSemiringTensorDot;
import org.flag4j.linalg.operations.dense.ring_ops.DenseRingTensorOps;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiRingElemMult;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringConversions;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringOperations;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;

/**
 * <p>The base class for all dense {@link Ring} tensors.</p>
 * <p>The {@link #entries} of an AbstractDenseRingTensor are mutable but the {@link #shape} is fixed.</p>
 *
 * @param <T> The type of this dense ring tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #toCoo()}) may result in a sparse tensor.
 * @param <V> The type of the {@link Ring} which this tensor's entries belong to.
 */
public abstract class AbstractDenseRingTensor<T extends AbstractDenseRingTensor<T, V>, V extends Ring<V>>
        extends AbstractTensor<T, Ring<V>[], V>
        implements RingTensorMixin<T, T, V> {

    /**
     * The zero element for the ring that this tensor's elements belong to.
     */
    protected Ring<V> zeroElement;

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractDenseRingTensor(Shape shape, Ring<V>[] entries) {
        super(shape, entries);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        this.zeroElement = (entries.length > 0) ? entries[0].getZero() : null;
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero entries of the COO tensor.
     * @param rowIndices Non-zero row indices of the COO tensor.
     * @param colIndices Non-zero column indices of the COO tensor.
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    protected abstract AbstractTensor<?, Ring<V>[], V> makeLikeCooTensor(
            Shape shape, Ring<V>[] entries, int[][] indices);


    /**
     * Gets the shape of this tensor.
     *
     * @return The shape of this tensor.
     */
    @Override
    public Shape getShape() {
        return super.getShape();
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
    public V get(int... indices) {
        return (V) entries[shape.getFlatIndex(indices)];
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
        entries[shape.getFlatIndex(indices)] = value;
        return (T) this;
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
        return makeLikeTensor(new Shape(shape.totalEntriesIntValueExact()), entries.clone());
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
        // No need to make explicit broadcastable check as the constructor will verify that the number of entries in the shape matches
        // the number of entries in the array.
        return makeLikeTensor(newShape, entries.clone());
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
        Ring<V>[] dest = new Ring[entries.length];
        TransposeDispatcher.dispatchTensor(entries, shape, axis1, axis2, dest);
        return makeLikeTensor(shape, (V[]) dest);
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
        Ring<V>[] dest = new Ring[entries.length];
        TransposeDispatcher.dispatchTensor(entries, shape, axes, dest);
        return makeLikeTensor(shape.permuteAxes(axes), (V[]) dest);
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
        Ring<V>[] sum = new Ring[entries.length];
        DenseSemiringOperations.add(entries, shape, b.entries, b.shape, sum);
        return makeLikeTensor(shape, sum);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     */
    public void addEq(T b) {
        DenseSemiringOperations.add(entries, shape, b.entries, b.shape, entries);
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
        Ring<V>[] prod = new Ring[entries.length];
        DenseSemiRingElemMult.dispatch(entries, shape, b.entries, b.shape, prod);
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
        DenseSemiringTensorDot<V> dot = new DenseSemiringTensorDot(shape, entries, src2.shape, src2.entries, aAxes, bAxes);
        Ring<V>[] dest = new Ring[dot.getOutputSize()];
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
        Ring<V>[] destEntries = new Ring[destShape.totalEntriesIntValueExact()];
        return makeLikeTensor(destShape, destEntries);
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
        Ring<V>[] diff = new Ring[entries.length];
        DenseRingTensorOps.sub(shape, entries, b.shape, b.entries, diff);
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
        DenseRingTensorOps.sub(shape, entries, b.shape, b.entries, entries);
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
        Ring<V>[] dest = new Ring[entries.length];
        TransposeDispatcher.dispatchTensorHermitian(shape, entries, axis1, axis2, dest);
        return makeLikeTensor(shape, dest);
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
        Ring<V>[] dest = new Ring[entries.length];
        TransposeDispatcher.dispatchTensorHermitian(shape, entries, axes, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return shape.getNdIndices(CompareRing.argminAbs(entries));
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return shape.getNdIndices(CompareRing.argmaxAbs(entries));
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return CompareRing.minAbs(entries);
    }


    /**
     * Finds the maximum absolute value in this tensor.
     *
     * @return The maximum absolute value in this tensor.
     */
    @Override
    public double maxAbs() {
        return CompareRing.maxAbs(entries);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public AbstractTensor<?, Ring<V>[], V> toCoo() {
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
    public AbstractTensor<?, Ring<V>[], V> toCoo(double estimatedSparsity) {
        SparseTensorData<Semiring<V>> data = DenseSemiringConversions.toCooTensor(shape, entries, estimatedSparsity);
        Ring<V>[] cooEntries = data.entries().toArray(new Ring[data.entries().size()]);
        return makeLikeCooTensor(data.shape(), cooEntries, data.indicesToArray());
    }
}
