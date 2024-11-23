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
import org.flag4j.linalg.operations.common.semiring_ops.CompareSemiring;
import org.flag4j.linalg.operations.sparse.SparseElementSearch;
import org.flag4j.linalg.operations.sparse.SparseUtils;
import org.flag4j.linalg.operations.sparse.coo.*;
import org.flag4j.linalg.operations.sparse.coo.semiring_ops.CooSemiringTensorOps;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;

/**
 * <p>Base class for all sparse tensors stored in coordinate list (COO) format. The entries of this COO tensor are elements of a
 * {@link Semiring}
 *
 * <p>The {@link #entries non-zero entries} and {@link #indices non-zero indices} of a COO tensor are mutable but the {@link #shape}
 * and total {@link #nnz number of non-zero entries} is fixed.
 *
 * <p>Sparse tensors allow for the efficient storage of and operations on tensors that contain many zero values.
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #entries} of the tensor. All other entries in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #entries}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many operations assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero entries in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code entries[i]}.
 *     </li>
 * </ul>
 *
 * @param <T> Type of this sparse COO tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #tensorDot(AbstractCooSemiringTensor, int[], int[])} ) between two sparse tensors results in a dense tensor.
 * @param <V> Type of the {@link Semiring} which the entries of this tensor belong to.
 */
public abstract class AbstractCooSemiringTensor<T extends AbstractCooSemiringTensor<T, U, V>,
        U extends AbstractDenseSemiringTensor<U, V>, V extends Semiring<V>>
        extends AbstractTensor<T, Semiring<V>[], V>
        implements SemiringTensorMixin<T, T, V> {

    /**
     * The zero element for the semiring that this tensor's elements belong to.
     */
    private Semiring<V> zeroElement;
    /**
     * <p>The non-zero indices of this sparse tensor.
     *
     * <p>Has shape {@code (nnz, rank)} where {@code nnz} is the number of non-zero entries in this sparse tensor.
     */
    public final int[][] indices;
    /**
     * The number of non-zero entries in this sparse tensor.
     */
    public final int nnz;
    /**
     * Stores the sparsity of this matrix.
     */
    public final double sparsity;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor. If this tensor is dense, this specifies all entries within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractCooSemiringTensor(Shape shape, Semiring<V>[] entries, int[][] indices) {
        super(shape, entries);
        if(indices.length != 0) ValidateParameters.ensureLengthEqualsRank(shape, indices[0].length);
        ValidateParameters.ensureArrayLengthsEq(entries.length, indices.length);
        ValidateParameters.validateTensorIndices(shape, indices);
        this.indices = indices;
        this.nnz = entries.length;
        sparsity = BigDecimal.valueOf(nnz).divide(new BigDecimal(shape.totalEntries())).doubleValue();

        // Attempt to set the zero element for the semiring.
        this.zeroElement = (entries.length > 0) ? entries[0].getZero() : null;
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero entries.
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero entries of the tensor to construct.
     * @param indices Indices of the non-zero entries of the tensor.
     * @return A tensor of the same type as this tensor with the specified shape and non-zero entries.
     */
    public abstract T makeLikeTensor(Shape shape, V[] entries, int[][] indices);


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero entries.
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero entries of the tensor to construct.
     * @param indices Indices of the non-zero entries of the tensor.
     * @return A tensor of the same type as this tensor with the specified shape and non-zero entries.
     */
    public abstract T makeLikeTensor(Shape shape, List<Semiring<V>> entries, List<int[]> indices);


    /**
     * Constructs a dense tensor that is a similar type as this sparse COO tensor.
     * @param shape Shape of the tensor to construct.
     * @param entries The entries of the dense tensor to construct.
     * @return A dense tensor that is a similar type as this sparse COO tensor.
     */
    public abstract U makeLikeDenseTensor(Shape shape, Semiring<V>[] entries);


    /**
     * Gets the zero element for the field of this tensor.
     * @return The zero element for the field of this tensor. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Semiring)} then {@code null} will be returned.
     */
    public V getZeroElement() {
        return (V) zeroElement;
    }


    /**
     * Sets the zero element for the field of this tensor.
     * @param zeroElement The zero element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the semiring.
     */
    public void setZeroElement(Semiring<V> zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }

    /**
     * Gets the sparsity of this tensor as a decimal percentage.
     * That is, the percentage of entries in this tensor that are zero.
     * @return The sparsity of this tensor as a decimal percentage.
     * @see #density()
     */
    public double sparsity() {
        return sparsity;
    }


    /**
     * Gets the density of this tensor as a decimal percentage.
     * That is, the percentage of entries in this tensor that are non-zero.
     * @return The density of this tensor as a decimal percentage.
     * @see #sparsity()
     */
    public double density() {
        return 1.0 - sparsity;
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
        SparseTensorData<Semiring<V>> data = CooSemiringTensorOps.add(
                shape, entries, indices,
                b.shape, b.entries, b.indices
        );

        return makeLikeTensor(data.shape(), data.entries(), data.indices());
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
        SparseTensorData<Semiring<V>> data = CooSemiringTensorOps.elemMult(
                shape, entries, indices, b.shape, b.entries, b.indices);
        return makeLikeTensor(data.shape(), data.entries(), data.indices());
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
    public U tensorDot(T src2, int[] aAxes, int[] bAxes) {
        CooTensorDot<V> problem = new CooTensorDot<>(shape, entries, indices,
                src2.shape, src2.entries, src2.indices,
                aAxes, bAxes);
        Semiring<V>[] dest = new Semiring[problem.getOutputSize()];
        problem.compute(dest);
        return makeLikeDenseTensor(problem.getOutputShape(), dest);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
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
        SparseTensorData<Semiring<V>> data = CooSemiringTensorOps.tensorTr(
                shape, entries, indices, axis1, axis2);
        return makeLikeTensor(data.shape(), data.entries(), data.indices());
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public T T() {
        Semiring<V>[] destEntries = new Semiring[nnz];
        int[][] destIndices = new int[nnz][rank];
        CooTranspose.tensorTranspose(shape, entries, indices,0, shape.getRank()-1, destEntries, destIndices);
        return makeLikeTensor(shape.swapAxes(0, rank-1), (V[]) destEntries, destIndices);
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
        Semiring<V>[] destEntries = new Semiring[nnz];
        int[][] destIndices = new int[nnz][rank];
        CooTranspose.tensorTranspose(shape, entries, indices, axis1, axis2, destEntries, destIndices);
        return makeLikeTensor(shape.swapAxes(axis1, axis2), (V[]) destEntries, destIndices);
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
        Semiring<V>[] destEntries = new Semiring[nnz];
        int[][] destIndices = new int[nnz][rank];
        CooTranspose.tensorTranspose(shape, entries, indices, axes, destEntries, destIndices);
        return makeLikeTensor(shape.permuteAxes(axes), (V[]) destEntries, destIndices);
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
     * Finds the minimum (non-zero) value in this tensor. If this tensor is complex, then this method finds the smallest value in
     * magnitude.
     *
     * @return The minimum (non-zero) value in this tensor.
     */
    @Override
    public V min() {
        return (V) CompareSemiring.min(entries);
    }


    /**
     * Finds the maximum (non-zero) value in this tensor.
     *
     * @return The maximum (non-zero) value in this tensor.
     */
    @Override
    public V max() {
        return (V) CompareSemiring.max(entries);
    }


    /**
     * Finds the indices of the minimum (non-zero) value in this tensor.
     *
     * @return The indices of the minimum (non-zero) value in this tensor.
     */
    @Override
    public int[] argmin() {
        return indices[CompareSemiring.argmin(entries)];
    }


    /**
     * Finds the indices of the maximum (non-zero) value in this tensor.
     *
     * @return The indices of the maximum (non-zero) value in this tensor.
     */
    @Override
    public int[] argmax() {
        return indices[CompareSemiring.argmin(entries)];
    }


    /**
     * Gets the element of this tensor at the specified target.
     *
     * @param target Index of the element to get.
     *
     * @return The element of this tensor at the specified index. If there is a non-zero value with the specified index, that value
     * will be returned. If there is no non-zero value at the specified index than the zero element will attempt to be
     * returned (i.e. the additive identity of the semiring). However, if the zero element could not be determined during
     * construction or if it was not set with {@link #setZeroElement(Semiring)} then
     * {@code null} will be returned.
     *
     * @throws ArrayIndexOutOfBoundsException If any target are not within this tensor.
     */
    @Override
    public V get(int... target) {
        ValidateParameters.validateTensorIndex(shape, target);
        V value = (V) CooGetSet.getCoo(entries, indices, target);
        return (value == null) ? getZeroElement() : value;
    }


    /**
     * Sets the element of this tensor at the specified target.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param target Index of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code target} is not within the bounds of this tensor.
     */
    @Override
    public T set(V value, int... target) {
        ValidateParameters.validateTensorIndex(shape, target);
        int idx = SparseElementSearch.binarySearchCoo(indices, target);

        Semiring<V>[] destEntries;
        int[][] destIndices;

        if (idx >= 0) {
            // Target index found.
            destEntries = entries.clone();
            destIndices = ArrayUtils.deepCopy(indices, null);
            destEntries[idx] = value;
            destIndices[idx] = target;
        } else {
            // Target not found, insert new value and index.
            destEntries = new Semiring[nnz + 1];
            destIndices = new int[nnz + 1][rank];
            int insertionPoint = - (idx + 1);
            CooGetSet.cooInsertNewValue(value, target, entries, indices, insertionPoint, destEntries, destIndices);
        }

        return makeLikeTensor(shape, (V[]) destEntries, destIndices);
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
        return makeLikeTensor(
                new Shape(shape.totalEntriesIntValueExact()),
                (V[]) entries.clone(),
                SparseUtils.cooFlattenIndices(shape, indices));
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
        int[] destShape = new int[indices[0].length];
        Arrays.fill(destShape, 1);
        destShape[axis] = shape.totalEntries().intValueExact();

        return makeLikeTensor(
                new Shape(destShape),
                (V[]) entries.clone(),
                SparseUtils.cooFlattenIndices(shape, indices, axis));
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
        return makeLikeTensor(newShape, (V[]) entries.clone(), SparseUtils.cooReshape(shape, newShape, indices));
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        CooDataSorter.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Converts this COO tensor to an equivalent dense tensor.
     * @return A dense tensor which is equivalent to this COO tensor.
     * @throws ArithmeticException If the number of entries in the dense tensor exceeds 2,147,483,647.
     */
    public U toDense() {
        Semiring<V>[] denseEntries = new Semiring[shape.totalEntriesIntValueExact()];
        CooConversions.toDense(shape, entries, indices, denseEntries);
        return makeLikeDenseTensor(shape, denseEntries);
    }
}
