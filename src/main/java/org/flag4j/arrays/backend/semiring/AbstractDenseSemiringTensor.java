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

package org.flag4j.arrays.backend.semiring;

import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.backend.SparseTensorData;
import org.flag4j.linalg.operations.TransposeDispatcher;
import org.flag4j.linalg.operations.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.operations.dense.DenseSemiringTensorDot;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiRingElemMult;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringConversions;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringOperations;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;


/**
 * <p>The base class for all dense {@link Semiring} tensors.</p>
 * <p>The {@link #entries} of an AbstractDenseSemiringTensor are mutable but the {@link #shape} is fixed.</p>
 *
 * @param <T> The type of this dense semi-ring tensor.
 * @param <U> Type of sparse tensor equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #toCoo()}) may result in a sparse tensor.
 * @param <V> The type of the {@link Semiring} which this tensor's entries belong to.
 */
public abstract class AbstractDenseSemiringTensor<T extends AbstractDenseSemiringTensor<T, V>, V extends Semiring<V>>
        extends AbstractTensor<T, Semiring<V>[], V>
        implements SemiringTensorMixin<T, T, V> {

    /**
     * The zero element for the semi-ring that this tensor's elements belong to.
     */
    private final Semiring<V> ZERO_ELEMENT;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractDenseSemiringTensor(Shape shape, Semiring<V>[] entries) {
        super(shape, entries);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        this.ZERO_ELEMENT = (entries.length > 0) ? entries[0].getZero() : null;
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero entries of the COO tensor.
     * @param rowIndices Non-zero row indices of the COO tensor.
     * @param colIndices Non-zero column indices of the COO tensor.
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    protected abstract AbstractTensor<?, Semiring<V>[], V> makeLikeCooTensor(
            Shape shape, Semiring<V>[] entries, int[][] indices);


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws IndexOutOfBoundsException If any indices are not within this tensor.
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
     * Flattens a tensor along the specified axis.
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
        // No need to make explicit broadcastable check as the constructor should verify that the number of entries in the shape
        // matches the number of entries in the array.
        return makeLikeTensor(newShape, entries.clone());
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
        Semiring<V>[] sum = new Semiring[entries.length];
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
        Semiring<V>[] prod = new Semiring[entries.length];
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
        Semiring<V>[] dest = new Semiring[dot.getOutputSize()];
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
        Semiring<V>[] destEntries = new Semiring[destShape.totalEntriesIntValueExact()];
        DenseSemiringOperations.tensorTr(shape, entries, axis1, axis2, destShape, destEntries);
        return makeLikeTensor(destShape, destEntries);
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    public V min() {
        return (V) CompareSemiring.min(entries);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    public V max() {
        return (V) CompareSemiring.max(entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    public int[] argmin() {
        return shape.getNdIndices(CompareSemiring.argmin(entries));
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    public int[] argmax() {
        return shape.getNdIndices(CompareSemiring.argmax(entries));
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
        Semiring<V>[] dest = new Semiring[entries.length];
        TransposeDispatcher.dispatchTensor(entries, shape, axis1, axis2, dest);
        return makeLikeTensor(shape, dest);
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
        Semiring<V>[] dest = new Semiring[entries.length];
        TransposeDispatcher.dispatchTensor(entries, shape, axes, dest);
        return makeLikeTensor(shape.permuteAxes(axes), dest);
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
     * <p>Checks if an object is equal to this tensor.</p>
     * <p>An object is considered equal to this tensor if it meets <i>all</i> the following conditions:
     * <ul>
     *     <li>The object is an instance of {@link AbstractDenseSemiringMatrix}.</li>
     *     <li>The object is not null: {@code object != null}</li>
     *     <li>The object has the same shape as this tensor: {@code ((AbstractDenseSemiringMatrix<?, ?, ?>) object).shape.equals(this
     *     .shape)}.</li>
     *     <li>The objects entries are element-wise equal to the entries of this tensor.</li>
     * </ul>
     * </p>
     * <p>These conditions implement an equivalency relation on non-null dense semi-ring tensors meaning the following are satisfied:
     * The {@code equals} method implements an equivalence relation
     * on non-null object references:
     * <ul>
     * <li>It is <i>reflexive</i>: for any non-null reference value
     *     {@code x}, {@code x.equals(x)} should return
     *     {@code true}.
     * <li>It is <i>symmetric</i>: for any non-null reference values
     *     {@code x} and {@code y}, {@code x.equals(y)}
     *     should return {@code true} if and only if
     *     {@code y.equals(x)} returns {@code true}.
     * <li>It is <i>transitive</i>: for any non-null reference values
     *     {@code x}, {@code y}, and {@code z}, if
     *     {@code x.equals(y)} returns {@code true} and
     *     {@code y.equals(z)} returns {@code true}, then
     *     {@code x.equals(z)} should return {@code true}.
     * <li>It is <i>consistent</i>: for any non-null reference values
     *     {@code x} and {@code y}, multiple invocations of
     *     {@code x.equals(y)} consistently return {@code true}
     *     or consistently return {@code false}, provided no
     *     information used in {@code equals} comparisons on the
     *     objects is modified.
     * <li>For any non-null reference value {@code x},
     *     {@code x.equals(null)} should return {@code false}.
     * </ul>
     * </p>
     * @param object Object to compare to this tensor.
     * @return {@code true} if this object is non-null and equal to this tensor as defined above; {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        AbstractDenseSemiringMatrix<?, ?, ?> src2 = (AbstractDenseSemiringMatrix<?, ?, ?>) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    /**
     * {@return a hash code value for this tensor} This method is
     * supported for the benefit of hash tables such as those provided by
     * {@link java.util.HashMap}.
     * <p>
     * The general contract of {@code hashCode} is:
     * <ul>
     * <li>Whenever it is invoked on the same object more than once during
     *     an execution of a Java application, the {@code hashCode} method
     *     must consistently return the same integer, provided no information
     *     used in {@code equals} comparisons on the object is modified.
     *     This integer need not remain consistent from one execution of an
     *     application to another execution of the same application.
     * <li>If two objects are equal according to the {@link
     *     #equals(Object) equals} method, then calling the {@code
     *     hashCode} method on each of the two objects must produce the
     *     same integer result.
     * <li>It is <em>not</em> required that if two objects are unequal
     *     according to the {@link #equals(Object) equals} method, then
     *     calling the {@code hashCode} method on each of the two objects
     *     must produce distinct integer results.  However, the programmer
     *     should be aware that producing distinct integer results for
     *     unequal objects may improve the performance of hash tables.
     * </ul>
     */
    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public AbstractTensor<?, Semiring<V>[], V> toCoo() {
        return toCoo(0.9);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo()
     */
    public AbstractTensor<?, Semiring<V>[], V> toCoo(double estimatedSparsity) {
        SparseTensorData<Semiring<V>> data = DenseSemiringConversions.toCooTensor(shape, entries, estimatedSparsity);
        Semiring<V>[] cooEntries = data.entries().toArray(new Semiring[data.entries().size()]);
        return makeLikeCooTensor(data.shape(), cooEntries, data.indicesToArray());
    }
}
