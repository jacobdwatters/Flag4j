/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.arrays_old.sparse;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.core.VectorMixin;
import org.flag4j.core.sparse_base.RealSparseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations_old.common.complex.ComplexOperations;
import org.flag4j.operations_old.common.real.RealOperations;
import org.flag4j.operations_old.dense.real.RealDenseTranspose;
import org.flag4j.operations_old.dense_sparse.coo.real.RealDenseSparseVectorOperations;
import org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.flag4j.operations_old.sparse.coo.SparseDataWrapper;
import org.flag4j.operations_old.sparse.coo.real.RealSparseEquals;
import org.flag4j.operations_old.sparse.coo.real.RealSparseVectorOperations;
import org.flag4j.operations_old.sparse.coo.real_complex.RealComplexSparseVectorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Real sparse vector stored in coordinate (COO) format.
 */
@Deprecated
public class CooVectorOld
        extends RealSparseTensorBase<CooVectorOld, VectorOld, CooCVectorOld, CVectorOld>
        implements VectorMixin<CooVectorOld, VectorOld, CooVectorOld, CooCVectorOld, Double, CooMatrixOld, MatrixOld, CooCMatrixOld> {

    /**
     * The size of this vector. That is, the number of entries in this vector.
     */
    public final int size;
    /**
     * Indices of non-zero entries in this sparse vector.
     */
    public final int[] indices;


    /**
     * Creates a sparse vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     */
    public CooVectorOld(int size) {
        super(new Shape(size), 0, new double[0], new int[0][0]);
        this.size = size;
        this.indices = new int[0];
    }


    /**
     * Creates a sparse vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries. Indices are assumed to be sorted in lexicographical order but this
     *                is <b>not</b> enforced. However, many algorithms assume the indices to be sorted. If they are not,
     *                use
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays_old are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public CooVectorOld(int size, int[] nonZeroEntries, int[] indices) {
        super(new Shape(size), nonZeroEntries.length,
                ArrayUtils.asDouble(nonZeroEntries, null),
                RealDenseTranspose.standardIntMatrix(new int[][]{indices})
        );
        this.size = size;
        this.indices = indices;
    }


    /**
     * Creates a sparse vector of specified size along with non-zero entries and their indices.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector. Entries assumed to be sorted by indices
     *                       (not enforced).
     * @param indices Indices of the nonZero entries. Assumed to be sorted (not enforced).
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays_old are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public CooVectorOld(int size, double[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.standardIntMatrix(new int[][]{indices})
        );

        this.size = size;
        this.indices = indices;
    }


    /**
     * Constructs a sparse vector whose non-zero values, indices, and size are specified by another sparse vector.
     * @param a Sparse vector to copy
     */
    public CooVectorOld(CooVectorOld a) {
        super(a.shape,
                a.nonZeroEntries(),
                a.entries.clone(),
                new int[a.indices.length][1]
        );
        this.indices = a.indices.clone();
        this.size = a.size;
    }


    /**
     * Creates a sparse vector of specified size, non-zero entries, and non-zero indices.
     * @param size Full size, including zeros, of the sparse vector.
     * @param entries Non-zero entries of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooVectorOld(int size, List<Double> entries, List<Integer> indices) {
        super(new Shape(size),
                entries.size(),
                ArrayUtils.fromDoubleList(entries),
                new int[indices.size()][1]
        );

        this.indices = ArrayUtils.fromIntegerList(indices);
        this.size = size;
    }


    /**
     * Checks if an object is equal to this sparse COO vector.
     * @param object Object to compare this sparse COO vector to.
     * @return True if the object is a {@link CooVectorOld}, has the same shape as this vector, and is element-wise equal to this
     * vector.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooVectorOld src2 = (CooVectorOld) object;
        return RealSparseEquals.vectorEquals(this, src2);
    }


    /**
     * Sets an index of a copy of this vector to a specified value.
     * Creates a copy of this vector and sets an index to the specified value. Note, unlike the dense version of this
     * method, this <b>does not</b> affect this vector.
     *
     * @param value   Value to set.
     * @param indices The index of for which to set the value.
     * @return A copy of this vector with the specified index set to the specified value.
     * @throws IllegalArgumentException If the number of indices provided is greater than 1.
     * @throws IllegalArgumentException If the index is negative or larger than the total vector size.
     */
    @Override
    public CooVectorOld set(double value, int... indices) {
        ParameterChecks.ensureEquals(indices.length, 1);
        ParameterChecks.ensureInRange(indices[0], 0, size, "index");

        int idx = Arrays.binarySearch(this.indices, indices[0]);
        double[] destEntries;
        int[] destIndices;

        if(idx >= 0) {
            // Then the index was found in the sparse vector.
            destIndices = this.indices.clone();
            destEntries = entries.clone();
            destEntries[idx] = value;

        } else{
            // Then the index was not found in the sparse vector.
            destIndices = new int[this.indices.length+1];
            destEntries = new double[entries.length+1];
            idx = -(idx+1);

            System.arraycopy(this.indices, 0, destIndices, 0, idx);
            destIndices[idx] = indices[0];
            System.arraycopy(this.indices, idx, destIndices, idx+1, this.indices.length-idx);

            System.arraycopy(entries, 0, destEntries, 0, idx);
            destEntries[idx] = value;
            System.arraycopy(entries, idx, destEntries, idx+1, entries.length-idx);
        }

        return new CooVectorOld(size, destEntries, destIndices);
    }


    /**
     * Copies and reshapes tensor if possible. The total number of entries in this tensor must match the total number of entries
     * in the reshaped tensor.
     *
     * @param shape Shape of the new tensor.
     *
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     *
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public CooVectorOld reshape(Shape shape) {
        // TODO: This should return a tensor. This would allow for a matrix or vector to be reshaped to any rank.
        ParameterChecks.ensureRank(1, shape);
        ParameterChecks.ensureBroadcastable(this.shape, shape);
        return copy();
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public CooVectorOld flatten() {
        return copy();
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public VectorOld join(VectorOld b) {
        double[] newEntries = new double[this.size + b.entries.length];

        // Copy over sparse values.
        for(int i=0; i<this.entries.length; i++) {
            newEntries[indices[i]] = entries[i];
        }

        // Copy over dense values.
        System.arraycopy(b.entries, 0, newEntries, this.size, b.entries.length);

        return new VectorOld(newEntries);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(CVectorOld b) {
        CNumber[] newEntries = new CNumber[this.size + b.entries.length];
        Arrays.fill(newEntries, CNumber.ZERO);

        // Copy over sparse values.
        for(int i=0; i<this.entries.length; i++) {
            newEntries[indices[i]] = new CNumber(entries[i]);
        }

        // Copy over dense values.
        System.arraycopy(b.entries, 0, newEntries, this.size, b.entries.length);

        return new CVectorOld(newEntries);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CooVectorOld join(CooVectorOld b) {
        double[] newEntries = new double[this.entries.length + b.entries.length];
        int[] newIndices = new int[this.indices.length + b.indices.length];

        // Copy values from this vector.
        System.arraycopy(this.entries, 0, newEntries, 0, this.entries.length);
        // Copy values from vector b.
        System.arraycopy(b.entries, 0, newEntries, this.entries.length, b.entries.length);

        // Copy indices from this vector.
        System.arraycopy(this.indices, 0, newIndices, 0, this.entries.length);

        // Copy the indices from vector b with a shift.
        for(int i=0; i<b.indices.length; i++) {
            newIndices[this.indices.length+i] = b.indices[i] + this.size;
        }

        return new CooVectorOld(this.size + b.size, newEntries, newIndices);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CooCVectorOld join(CooCVectorOld b) {
        CNumber[] newEntries = new CNumber[this.entries.length + b.entries.length];
        int[] newIndices = new int[this.indices.length + b.indices.length];

        // Copy values from this vector.
        ArrayUtils.arraycopy(this.entries, 0, newEntries, 0, this.entries.length);
        // Copy values from vector b.
        System.arraycopy(b.entries, 0, newEntries, this.entries.length, b.entries.length);

        // Copy indices from this vector.
        System.arraycopy(this.indices, 0, newIndices, 0, this.entries.length);

        // Copy the indices from vector b with a shift.
        for(int i=0; i<b.indices.length; i++) {
            newIndices[this.indices.length+i] = b.indices[i] + this.size;
        }

        return new CooCVectorOld(this.size + b.size, newEntries, newIndices);
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public CooMatrixOld stack(VectorOld b) {
        ParameterChecks.ensureEqualShape(this.shape, b.shape);

        double[] destEntries = new double[nnz + b.length()];
        int[][] indices = new int[2][nnz + b.length()];

        // Copy sparse values and column indices (row indices will be implicitly zero)
        System.arraycopy(entries, 0, destEntries,0,  entries.length);
        System.arraycopy(this.indices, 0, indices[1], 0, this.indices.length);

        // Copy dense values. Set column indices as range and set row indices to 1.
        int[] rowIndices = new int[b.size];
        Arrays.fill(rowIndices, 1);
        System.arraycopy(b.entries, 0, destEntries, entries.length,  b.size);
        System.arraycopy(rowIndices, 0, indices[0], entries.length,  b.size);
        System.arraycopy(ArrayUtils.intRange(0, b.size), 0, indices[1], entries.length,  b.size);

        return new CooMatrixOld(2, b.size, destEntries, indices[0], indices[1]);
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public CooMatrixOld stack(CooVectorOld b) {
        ParameterChecks.ensureEqualShape(this.shape, b.shape);

        double[] entries = new double[this.entries.length + b.entries.length];
        int[][] indices = new int[2][this.indices.length + b.indices.length]; // Row and column indices.

        // Copy values from this vector.
        System.arraycopy(this.entries, 0, entries, 0, this.entries.length);
        // Copy values from vector b.
        System.arraycopy(b.entries, 0, entries, this.entries.length, b.entries.length);

        // Set row indices to 1 for b values (this vectors row indices are 0 which was implicitly set already).
        Arrays.fill(indices[0], this.indices.length, entries.length, 1);

        // Copy indices from this vector to the column indices.
        System.arraycopy(this.indices, 0, indices[1], 0, this.entries.length);
        // Copy indices from b vector to the column indices.
        System.arraycopy(b.indices, 0, indices[1], this.entries.length, b.entries.length);

        return new CooMatrixOld(new Shape(2, this.size), entries, indices[0], indices[1]);
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public CooCMatrixOld stack(CVectorOld b) {
        ParameterChecks.ensureEqualShape(this.shape, b.shape);

        CNumber[] destEntries = new CNumber[nnz + b.length()];
        int[][] indices = new int[2][nnz + b.length()];

        // Copy sparse values and column indices (row indices will be implicitly zero)
        ArrayUtils.arraycopy(entries, 0, destEntries,0,  entries.length);
        System.arraycopy(this.indices, 0, indices[1], 0, this.indices.length);

        // Copy dense values. Set column indices as range and set row indices to 1.
        int[] rowIndices = new int[b.size];
        Arrays.fill(rowIndices, 1);
        System.arraycopy(b.entries, 0, destEntries, entries.length,  b.size);
        System.arraycopy(rowIndices, 0, indices[0], entries.length,  b.size);
        System.arraycopy(ArrayUtils.intRange(0, b.size), 0, indices[1], entries.length,  b.size);

        return new CooCMatrixOld(2, b.size, destEntries, indices[0], indices[1]);
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public CooCMatrixOld stack(CooCVectorOld b) {
        ParameterChecks.ensureEqualShape(this.shape, b.shape);

        CNumber[] entries = new CNumber[this.entries.length + b.entries.length];
        int[][] indices = new int[2][this.indices.length + b.indices.length]; // Row and column indices.

        // Copy values from this vector.
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.entries.length);
        // Copy values from vector b.
        System.arraycopy(b.entries, 0, entries, this.entries.length, b.entries.length);

        // Set row indices to 1 for b values (this vectors row indices are 0 which was implicitly set already).
        Arrays.fill(indices[0], this.indices.length, entries.length, 1);

        // Copy indices from this vector to the column indices.
        System.arraycopy(this.indices, 0, indices[1], 0, this.entries.length);
        // Copy indices from b vector to the column indices.
        System.arraycopy(b.indices, 0, indices[1], this.entries.length, b.entries.length);

        return new CooCMatrixOld(new Shape(2, this.size), entries, indices[0], indices[1]);
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b    VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CooMatrixOld stack(VectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b    VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CooMatrixOld stack(CooVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b    VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CooCMatrixOld stack(CVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b    VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CooCMatrixOld stack(CooCVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public VectorOld add(VectorOld B) {
        return RealDenseSparseVectorOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooVectorOld add(CooVectorOld B) {
        return RealSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld add(CVectorOld B) {
        return RealComplexDenseSparseVectorOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CooCVectorOld add(CooCVectorOld B) {
        return RealComplexSparseVectorOperations.add(B, this);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public VectorOld sub(VectorOld B) {
        return RealDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public VectorOld add(double a) {
        return RealSparseVectorOperations.add(this, a);
    }


    /**
     * Adds specified value to all entries of this tensor.
     * If zeros are introduced, they will be explicitly stored.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVectorOld add(CNumber a) {
        return RealComplexSparseVectorOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooVectorOld sub(CooVectorOld B) {
        return RealSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld sub(CVectorOld B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CooCVectorOld sub(CooCVectorOld B) {
        return RealComplexSparseVectorOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public VectorOld sub(double a) {
        return RealSparseVectorOperations.sub(this, a);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVectorOld sub(CNumber a) {
        return RealComplexSparseVectorOperations.sub(this, a);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooVectorOld elemMult(VectorOld B) {
        return RealDenseSparseVectorOperations.elemMult(B, this);
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CooVectorOld div(double divisor) {
        return new CooVectorOld(size, RealOperations.scalDiv(entries, divisor), indices.clone());
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CooCVectorOld div(CNumber divisor) {
        return new CooCVectorOld(size, ComplexOperations.scalDiv(entries, divisor), indices.clone());
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}. Since a vector is a rank 1 tensor, this just
     *      * copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooVectorOld T() {
        return this.copy();
    }


    /**
     * Gets the element in this tensor at the specified indices. This sparse vectors indices are assumed to
     * be sorted lexicographically. If this is not the case call
     * {@link #sortIndices() this.sparseSort()} before calling this method.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.ensureEquals(indices.length, 1);
        ParameterChecks.ensureInRange(indices[0], 0, size, "index");

        int idx = Arrays.binarySearch(this.indices, indices[0]);
        return idx>=0 ? entries[idx] : 0;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CooVectorOld copy() {
        return new CooVectorOld(this.size, this.entries.clone(), this.indices.clone());
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooVectorOld elemMult(CooVectorOld B) {
        return RealSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooCVectorOld elemMult(CVectorOld B) {
        return RealComplexDenseSparseVectorOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooCVectorOld elemMult(CooCVectorOld B) {
        return RealComplexSparseVectorOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param b VectorOld to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code b} do not have the same size.
     */
    @Override
    public CooVectorOld elemDiv(VectorOld b) {
        return RealDenseSparseVectorOperations.elemDiv(this, b);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param b VectorOld to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code b} do not have the same size.
     */
    @Override
    public CooCVectorOld elemDiv(CVectorOld b) {
        return RealComplexDenseSparseVectorOperations.elemDiv(this, b);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public Double inner(VectorOld b) {
        ParameterChecks.ensureEqualShape(shape, b.shape);
        return RealDenseSparseVectorOperations.inner(b.entries, this.entries, this.indices, this.size);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public Double inner(CooVectorOld b) {
        return RealSparseVectorOperations.inner(this, b);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector.
     */
    @Override
    public CooVectorOld normalize() {
        double norm = VectorNorms.norm(this);
        return norm==0 ? new CooVectorOld(size) : this.div(norm);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CVectorOld b) {
        return RealComplexDenseSparseVectorOperations.inner(this.entries, this.indices, this.size, b.entries);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CooCVectorOld b) {
        return RealComplexSparseVectorOperations.inner(this, b);
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public MatrixOld outer(VectorOld b) {
        return new MatrixOld(
                this.size,
                b.size,
                RealDenseSparseVectorOperations.outerProduct(this.entries, this.indices, this.size, b.entries)
        );
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public MatrixOld outer(CooVectorOld b) {
        return RealSparseVectorOperations.outerProduct(this, b);
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrixOld outer(CVectorOld b) {
        return new CMatrixOld(
                this.size,
                b.size,
                RealComplexDenseSparseVectorOperations.outerProduct(this.entries, this.indices, this.size, b.entries)
        );
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrixOld outer(CooCVectorOld b) {
        return RealComplexSparseVectorOperations.outerProduct(this, b);
    }


    /**
     * Checks if a vector is parallel to this vector. This sparse vectors indices are assumed to be sorted lexicographically.
     * If this is not the case call {@link #sortIndices() this.sparseSort()} before calling this method.
     *
     * @param b VectorOld to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isParallel(VectorOld b) {
        final double tol = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        boolean result;

        if(this.size!=b.size) {
            result = false;
        } else if(this.size<=1) {
            result = true;
        } else if(this.isZeros() || b.isZeros()) {
            result = true; // Any vector is parallel to zero vector.
        } else {
            result = true;
            int sparseIndex = 0;
            double scale = 0;

            // Find first non-zero entry in b and compute the scaling factor (we know there is at least one from else-if).
            for(int i=0; i<b.size; i++) {
                if(b.entries[i]!=0) {
                    scale = this.entries[i]/b.entries[this.indices[i]];
                    break;
                }
            }

            for(int i=0; i<b.size; i++) {
                if(sparseIndex >= this.entries.length || i!=this.indices[sparseIndex]) {
                    // Then this index is not in the sparse vector.
                    if(b.entries[i] != 0) {
                        result = false;
                        break;
                    }

                } else {
                    // Ensure the scaled entry is approximately equal to the entry in this vector.
                    if(Math.abs(b.entries[i]*scale - this.entries[sparseIndex]) > tol) {
                        result = false;
                        break;
                    }

                    sparseIndex++;
                }
            }
        }

        return result;
    }


    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b VectorOld to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isPerp(VectorOld b) {
        final double tol = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        boolean result;

        if(this.size!=b.size) {
            result = false;
        } else {
            result = Math.abs(this.inner(b)) < tol;
        }

        return result;
    }


    /**
     * Converts this vector to an equivalent matrix as if it were a column vector.
     *
     * @return A matrix equivalent to this vector as if the vector is a column vector.
     * @see #toMatrix(boolean)
     */
    @Override
    public CooMatrixOld toMatrix() {
        int[] rowIndices = indices.clone();
        int[] colIndices = new int[entries.length];

        return new CooMatrixOld(this.size, 1, entries.clone(), rowIndices, colIndices);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     *                    <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     * @see #toMatrix()
     */
    @Override
    public CooMatrixOld toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new CooMatrixOld(this.size, 1, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new CooMatrixOld(1, this.size, entries.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor which is equivalent to this vector.
     */
    public CooTensorOld toTensor() {
        return new CooTensorOld(
                this.shape,
                this.entries.clone(),
                RealDenseTranspose.standardIntMatrix(new int[][]{this.indices.clone()})
        );
    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n    The number of times to extend this vector. Must be a positive value.
     * @param axis Axis along which to extend. If {@code axis=0}, then the vector will be treated as a row vector. If
     *    {@code axis=1} then the vector will be treated as a column vector.
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public CooMatrixOld extend(int n, int axis) {
        ParameterChecks.ensureGreaterEq(1, n, "n");
        ParameterChecks.ensureAxis2D(axis);

        int[][] matIndices = new int[2][n*nnz];
        double[] matEntries = new double[n*nnz];
        Shape matShape;

        if(axis==0) {
            matShape = new Shape(n, this.size);
            int[] rowIndices = new int[indices.length];

            for(int i=0; i<n; i++) {
                Arrays.fill(rowIndices, i);
                System.arraycopy(entries, 0, matEntries, (n-1)*i, nnz);
                System.arraycopy(rowIndices, 0, matIndices[0], (n-1)*i, nnz);
                System.arraycopy(indices, 0, matIndices[1], (n-1)*i, nnz);
            }

        } else {
            matShape = new Shape(this.size, n);
            int[] rowIndices = new int[n];
            int[] colIndices = ArrayUtils.intRange(0, n);

            for(int i=0; i<entries.length; i++) {
                Arrays.fill(rowIndices, indices[i]);

                Arrays.fill(matEntries, (entries.length+1)*i, (entries.length+1)*i + n, entries[i]);
                System.arraycopy(rowIndices, 0, matIndices[0], (entries.length+1)*i, n);
                System.arraycopy(colIndices, 0, matIndices[1], (entries.length+1)*i, n);
            }
        }

        return new CooMatrixOld(matShape, matEntries, matIndices[0], matIndices[1]);
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return this.size;
    }


    /**
     * A factory for creating a real sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CooVectorOld makeTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooVectorOld(size, entries, RealDenseTranspose.standardIntMatrix(indices)[0]);
    }


    /**
     * A factory for creating a real dense tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected VectorOld makeDenseTensor(Shape shape, double[] entries) {
        return new VectorOld(entries);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CooCVectorOld makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CooCVectorOld(size, entries, RealDenseTranspose.standardIntMatrix(indices)[0]);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant tensor will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CooCVectorOld toComplex() {
        CNumber[] destEntries = new CNumber[entries.length];
        ArrayUtils.copy2CNumber(entries, destEntries);

        return new CooCVectorOld(
                this.size,
                destEntries,
                indices.clone()
        );
    }


    /**
     * Flattens a tensor along the specified axis. This simply copies the vector since it is rank 1.
     *
     * @param axis Axis along which to flatten tensor. Must be 0.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CooVectorOld flatten(int axis) {
        assert(axis==0) : "Axis must be zero but got " + axis;
        return this.copy();
    }


    /**
     * gets the size of this vector.
     *
     * @return The number of total entries (including zeros) of this vector.
     */
    @Override
    public int size() {
        return this.size;
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector. If {@code axis=0} then each row of the resulting matrix will be equivalent to
     * this vector. If {@code axis=1} then each column of the resulting matrix will be equivalent to this vector.
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    @Override
    public CooMatrixOld repeat(int n, int axis) {
        ParameterChecks.ensureInRange(axis, 0, 1, "axis");
        ParameterChecks.ensureGreaterEq(0, n, "n");

        Shape tiledShape;
        double[] tiledEntries = new double[n*entries.length];
        int[] tiledRows = new int[tiledEntries.length];
        int[] tiledCols = new int[tiledEntries.length];
        int nnz = entries.length;

        if(axis==0) {
            tiledShape = new Shape(n, size);

            for(int i=0; i<n; i++) { // Copy values into row and set col indices as vector indices.
                System.arraycopy(entries, 0, tiledEntries, i*nnz, nnz);
                System.arraycopy(indices, 0, tiledCols, i*nnz, indices.length);
                Arrays.fill(tiledRows, i*nnz, (i+1)*nnz, i);
            }
        } else {
            int[] colIndices = ArrayUtils.intRange(0, n);
            tiledShape = new Shape(size, n);

            for(int i=0; i<entries.length; i++) {
                Arrays.fill(tiledEntries, i*n, (i+1)*n, entries[i]);
                Arrays.fill(tiledRows, i*n, (i+1)*n, indices[i]);
                System.arraycopy(colIndices, 0, tiledCols, i*n, n);
            }
        }

        return new CooMatrixOld(tiledShape, tiledEntries, tiledRows, tiledCols);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CooVectorOld getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooVectorOld tensor, double relTol, double absTol) {
        return RealSparseEquals.allCloseVector(this, tensor, relTol, absTol);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public VectorOld toDense() {
        double[] entries = new double[size];

        for(int i = 0; i< nnz; i++) {
            entries[indices[i]] = this.entries[i];
        }

        return new VectorOld(entries);
    }


    /**
     * Creates a sparse tensor from a dense tensor.
     *
     * @param src Dense tensor to convert to a sparse tensor.
     * @return A sparse tensor which is equivalent to the {@code src} dense tensor.
     */
    public static CooVectorOld fromDense(VectorOld src) {
        List<Double> nonZeroEntries = new ArrayList<>((int) (src.entries.length*0.8));
        List<Integer> indices = new ArrayList<>((int) (src.entries.length*0.8));

        // Fill entries with non-zero values.
        for(int i=0; i<src.entries.length; i++) {
            if(src.entries[i] != 0d) {
                nonZeroEntries.add(src.entries[i]);
                indices.add(i);
            }
        }

        return new CooVectorOld(
                src.size,
                ArrayUtils.fromDoubleList(nonZeroEntries),
                ArrayUtils.fromIntegerList(indices)
        );
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flatten entries of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("Full Shape: %s\n", shape));
        result.append("Non-zero entries: [");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry now
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]\n");
        result.append("Indices: ").append(Arrays.toString(indices));

        return result.toString();
    }
}


