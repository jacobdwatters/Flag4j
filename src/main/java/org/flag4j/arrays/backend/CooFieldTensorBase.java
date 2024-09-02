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

package org.flag4j.arrays.backend;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.fields.RealFloat64;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooFieldTensor;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.operations.sparse.coo.SparseDataWrapper;
import org.flag4j.operations.sparse.coo.field_ops.CooFieldTensorDot;
import org.flag4j.operations.sparse.coo.field_ops.CooFieldTensorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * <p>Base class for all sparse tensors stored in coordinate list (COO) format. The entries of this COO tensor are elements of a
 * {@link Field}</p>
 *
 * <p>The non-zero entries and non-zero indices of a COO tensor are mutable but the shape and total number of non-zero entries is
 * fixed.</p>
 *
 * <p>Sparse tensors allow for the efficient storage of and operations on tensors that contain many zero values.</p>
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).</p>
 *
 * <p>A sparse COO tensor is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #entries} of the tensor. All other entries in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #entries}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many operations assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.</p>
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero entries in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code entries[i]}.</p>
 *     </li>
 * </ul>
 *
 * @param <T> Type of this sparse COO tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #tensorDot(TensorOverSemiRing, int)}) between two sparse tensors result in a dense tensor.
 * @param <V> Type of the {@link Field field} which the entries of this tensor belong to.
 */
public abstract class CooFieldTensorBase<T extends CooFieldTensorBase<T, U, V>,
        U extends DenseFieldTensorBase<U, T, V>, V extends Field<V>>
        extends FieldTensorBase<T, U, V> implements SparseTensorMixin<U, T> {

    /**
     * <p>The non-zero indices of this sparse tensor.</p>
     *
     * <p>Has shape {@code (nnz, rank)} where {@code nnz} is the number of non-zero entries in this sparse tensor.</p>
     */
    public final int[][] indices;
    /**
     * The number of non-zero entries in this sparse tensor.
     */
    public final int nnz;
    /**
     * Stores the sparsity of this matrix.
     */
    private double sparsity = -1.0;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor. If this tensor is dense, this specifies all entries within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public CooFieldTensorBase(Shape shape, V[] entries, int[][] indices) {
        super(shape, entries);
        ParameterChecks.ensureLengthEqualsRank(shape, indices[0].length);
        ParameterChecks.ensureArrayLengthsEq(entries.length, indices.length);

        this.indices = indices;
        this.nnz = entries.length;
    }


    /**
     * <p>Computes the element-wise reciprocals of the non-zero elements of this sparse tensor.</p>
     *
     * <p>Note: This method <b>only</b> computes the reciprocals of the non-zero elements.</p>
     *
     * @return A tensor containing the reciprocal non-zero elements of this tensor.
     */
    @Override
    public T recip() {
        /* This method overrides from FieldTensorBase to make clear it is only computing the
            multiplicative inverse for the non-zero elements of the tensor */
        Field<V>[] recip = new Field[entries.length];
        for(int i=0, size=entries.length; i<size; i++)
            recip[i] = entries[i].multInv();

        return makeLikeTensor(shape, (V[]) recip);
    }


    /**
     * <p>Adds a real value to each non-zero entry of this tensor.</p>
     *
     * <p>Note: this method <b>only</b> operates on the non-zero entries of this tensor.</p>
     *
     * @param b Value to add to each non-zero value of this tensor.
     *
     * @return Sum of this tensor with {@code b}.
     */
    @Override
    public T add(double b) {
        return super.add(b);
    }


    /**
     * <p>Subtracts a real value from each non-zero entry of this tensor.</p>
     *
     * <p>Note: this method <b>only</b> operates on the non-zero entries of this tensor.</p>
     *
     * @param b Value to subtract from each non-zero value of this tensor.
     *
     * @return Difference of this tensor with {@code b}.
     */
    @Override
    public T sub(double b) {
        return super.sub(b);
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(V b) {
        return super.sub(b);
    }


    /**
     * <p>Subtracts a scalar value from each non-zero entry of this tensor and stores the result in this tensor.</p>
     *
     * <p>Note: this method <b>only</b> operates on the non-zero entries of this tensor.</p>
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(V b) {
        super.subEq(b);
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
        return CooFieldTensorOperations.sub(this, b);
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
        int rank = getRank();
        ParameterChecks.ensureIndexInBounds(rank, axis1, axis2);

        if(axis1 == axis2) return copy(); // Simply return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        Field<V>[] transposeEntries = new Field[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i].conj();
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create sparse coo tensor and sort values lexicographically.
        T transpose = makeLikeTensor(shape.swapAxes(axis1, axis2), (V[]) transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
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
        int rank = getRank();
        ParameterChecks.ensureEquals(rank, axes.length);
        ParameterChecks.ensurePermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        Field<V>[] transposeEntries = new Field[nnz];

        // Permute the indices according to the permutation array.
        for(int i = 0; i < nnz; i++) {
            transposeEntries[i] = entries[i].conj();
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create sparse coo tensor and sort values lexicographically.
        T transpose = makeLikeTensor(shape.swapAxes(axes), (V[]) transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * <p>Adds a scalar value to each non-zero entry of this tensor.</p>
     *
     * <p>Note: this method <b>only</b> operates on the non-zero entries of this tensor.</p>
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(V b) {
        return super.add(b);
    }


    /**
     * <p>Adds a scalar value to each non-zero entry of this tensor and stores the result in this tensor.</p>
     *
     * <p>Note: this method <b>only</b> operates on the non-zero entries of this tensor.</p>
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(V b) {
        super.addEq(b);
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
        return CooFieldTensorOperations.add(this, b);
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
        return CooFieldTensorOperations.elemMult(this, b);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by
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
        // Validate parameters.
        ParameterChecks.ensureNotEquals(axis1, axis2);
        ParameterChecks.ensureValidIndices(getRank(), axis1, axis2);
        ParameterChecks.ensureEquals(shape.get(axis1), shape.get(axis2));

        int rank = getRank();
        int[] dims = shape.getDims();

        // Determine the shape of the resulting tensor.
        int[] traceShape = new int[rank - 2];
        int newShapeIndex = 0;
        for (int i = 0; i < rank; i++) {
            if (i != axis1 && i != axis2) {
                traceShape[newShapeIndex++] = dims[i];
            }
        }

        // Use a map to accumulate non-zero entries that are on the diagonal.
        Map<Integer, V> resultMap = new HashMap<>();
        int[] strides = shape.getStrides();

        // Iterate through the non-zero entries and accumulate trace for those on the diagonal.
        for (int i = 0; i < this.nnz; i++) {
            int[] indices = this.indices[i];
            V value = this.entries[i];

            // Check if the current entry is on the diagonal
            if (indices[axis1] == indices[axis2]) {
                // Compute a linear index for the resulting tensor by ignoring axis1 and axis2.
                int linearIndex = 0;
                int stride = 1;

                for (int j = rank - 1; j >= 0; j--) {
                    if (j != axis1 && j != axis2) {
                        linearIndex += indices[j] * stride;
                        stride *= dims[j];
                    }
                }

                // Accumulate the value in the result map.
                resultMap.put(linearIndex, resultMap.getOrDefault(linearIndex, getZeroElement()).add(value));
            }
        }

        // Construct the result tensor from the accumulated non-zero entries
        int resultNnz = resultMap.size();
        int[][] resultIndices = new int[resultNnz][rank - 2];
        Field<V>[] resultEntries = new Field[resultNnz];
        int resultIndex = 0;

        for (Map.Entry<Integer, V> entry : resultMap.entrySet()) {
            int linearIndex = entry.getKey();
            V entryValue = entry.getValue();

            // Use the getIndices method to convert the flat index to n-dimensional index.
            int[] multiDimIndices = shape.getIndices(linearIndex);

            // Copy relevant dimensions to resultIndices, excluding axis1 and axis2.
            int resultDimIndex = 0;
            for (int j = 0; j < rank; j++) {
                if (j != axis1 && j != axis2) {
                    resultIndices[resultIndex][resultDimIndex++] = multiDimIndices[j];
                }
            }

            resultEntries[resultIndex] = entryValue;
            resultIndex++;
        }

        return makeLikeTensor(new Shape(traceShape), (V[]) resultEntries, resultIndices);
    }


    /**
     * <p>Computes the product of all non-zero values in this tensor.</p>
     *
     * <p>NOTE: This is <b>only</b> the product of the non-zero values in this tensor.</p>
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public V prod() {
        // Overrides from FieldTensorBase to emphasize that the product is only for non-zero entries.
        return super.prod();
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices. If this tensor has zero non-zero indices, then {@code null} is
     * returned.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    @Override
    public V get(int... indices) {
        ParameterChecks.ensureValidIndex(shape, indices);
        if(entries.length == 0) return null; // Can not get reference of field so no way to get zero element.

        for(int i=0; i<nnz; i++)
            if(Arrays.equals(this.indices[i], indices)) return entries[i];

        return entries[0].getZero(); // Return zero if the index is not found
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
        int[][] destIndices = new int[entries.length][1];

        for(int i=0, size=entries.length; i<size; i++)
            destIndices[i][0] = shape.entriesIndex(indices[i]);

        return makeLikeTensor(new Shape(shape.totalEntries().intValueExact()), entries.clone(), destIndices);
    }


    /**
     * Flattens a tensor along the specified axis. This preserves the rank of the tensor.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ParameterChecks.ensureIndexInBounds(indices[0].length, axis);
        int[][] destIndices = new int[indices.length][indices[0].length];

        // Compute new shape.
        int[] destShape = new int[indices[0].length];
        Arrays.fill(destShape, 1);
        destShape[axis] = shape.totalEntries().intValueExact();

        for(int i=0, size=entries.length; i<size; i++)
            destIndices[i][axis] = shape.entriesIndex(indices[i]);

        return makeLikeTensor(new Shape(destShape), entries.clone(), destIndices);
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
        ParameterChecks.ensureBroadcastable(shape, newShape);

        int rank = indices[0].length;
        int newRank = newShape.getRank();
        int nnz = entries.length;

        int[] oldStrides = shape.getStrides();
        int[] newStrides = newShape.getStrides();

        int[][] newIndices = new int[nnz][newRank];

        for(int i=0; i<nnz; i++) {
            int[] idxRow = indices[i];
            int[] newIdxRow = newIndices[i];

            int flatIndex = 0;
            for(int j=0; j < rank; j++) {
                flatIndex += idxRow[j] * oldStrides[j];
            }

            for(int j=0; j<newRank; j++) {
                newIdxRow[j] = flatIndex / newStrides[j];
                flatIndex %= newStrides[j];
            }
        }

        return makeLikeTensor(newShape, entries.clone(), newIndices);
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Entries of the spares tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     */
    @Override
    public abstract T makeLikeTensor(Shape shape, V[] entries);


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    public abstract T makeLikeTensor(Shape shape, V[] entries, int[][] indices);


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    public abstract T makeLikeTensor(Shape shape, List<V> entries, List<int[]> indices);


    /**
     * Finds the minimum non-zero value in this tensor. If this tensor is complex, then this method finds the smallest value in
     * magnitude.
     *
     * @return The minimum non-zero value (smallest in magnitude for a complex valued tensor) in this tensor. If this tensor does
     * not have any non-zero values, then {@code null} will be returned.
     */
    @Override
    public V min() {
        // Overrides method in FieldTensorBase to emphasize that the method works on the non-zero elements only.
        return super.min();
    }


    /**
     * Finds the maximum non-zero value in this tensor. If this tensor is complex, then this method finds the largest value in
     * magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor. If this tensor does not have any
     * non-zero values, then {@code null} will be returned.
     */
    @Override
    public V max() {
        // Overrides method in FieldTensorBase to emphasize that the method works on the non-zero elements only.
        return super.max();
    }


    /**
     * Finds the minimum non-zero value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        // Overrides method in FieldTensorBase to emphasize that the method works on the non-zero elements only.
        return super.minAbs();
    }


    /**
     * Finds the maximum non-zero value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        // Overrides method in FieldTensorBase to emphasize that the method works on the non-zero elements only.
        return super.maxAbs();
    }


    /**
     * Finds the indices of the minimum non-zero value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values then an empty is returned.
     */
    @Override
    public int[] argmin() {
        if(nnz > 0) return indices[CompareField.argmin(entries)];
        else return new int[0];
    }


    /**
     * Finds the indices of the maximum non-zero value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values then an empty array is returned.
     */
    @Override
    public int[] argmax() {
        if(nnz > 0) return indices[CompareField.argmax(entries)];
        else return new int[0];
    }


    /**
     * Finds the indices of the minimum absolute non-zero value in this tensor.
     *
     * @return The indices of the minimum absolute non-zero value in this tensor. If this value occurs multiple times, the indices of
     * the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values then an empty array is returned.
     */
    @Override
    public int[] argminAbs() {
        if(nnz > 0) return indices[CompareField.argminAbs(entries)];
        else return new int[0];
    }


    /**
     * Finds the indices of the maximum absolute non-zero value in this tensor.
     *
     * @return The indices of the maximum absolute non-zero value in this tensor. If this value occurs multiple times, the indices of
     * the
     * first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values then an empty array is returned.
     */
    @Override
    public int[] argmaxAbs() {
        if(nnz > 0) return indices[CompareField.argmaxAbs(entries)];
        else return new int[0];
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
        // This cast should be safe because FieldTensor<T> extends FieldTensorBase<FieldTensor<T>, FieldTensor<T>, T>
        //  and tensorDot(...) returns FieldTensorBase<U, U, V>.
        return (U) CooFieldTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
        int rank = getRank();
        ParameterChecks.ensureIndexInBounds(rank, axis1, axis2);

        if(axis1 == axis2) return copy(); // Simply return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        Field<V>[] transposeEntries = new Field[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i];
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create sparse coo tensor and sort values lexicographically by indices.
        T transpose = makeLikeTensor(shape.swapAxes(axis1, axis2), (V[]) transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
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
        int rank = getRank();
        ParameterChecks.ensureEquals(rank, axes.length);
        ParameterChecks.ensurePermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        Field<V>[] transposeEntries = new Field[nnz];

        // Permute the indices according to the permutation array.
        for(int i=0; i < nnz; i++) {
            transposeEntries[i] = entries[i];
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create sparse COO tensor and sort values lexicographically by indices.
        T transpose = makeLikeTensor(shape.swapAxes(axes), (V[]) transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, entries.clone(), ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Makes a dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     * @return A dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     */
    public abstract U makeDenseTensor(Shape shape, V[] entries);


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CooFieldTensor<RealFloat64> abs() {
        RealFloat64[] abs = new RealFloat64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            abs[i] = new RealFloat64(entries[i].abs());

        return new CooFieldTensor<RealFloat64>(shape, abs, ArrayUtils.deepCopy(indices, null));
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
     * @return The density of this sparse tensor.
     */
    @Override
    public double sparsity() {
        if(sparsity == -1) { // Compute sparsity if needed.
            BigInteger totalEntries = totalEntries();
            BigDecimal sparsity = new BigDecimal(totalEntries).subtract(BigDecimal.valueOf(nnz));
            sparsity = sparsity.divide(new BigDecimal(totalEntries), 50, RoundingMode.HALF_UP);
            this.sparsity = sparsity.doubleValue();
        }

        return sparsity;
    }
}
