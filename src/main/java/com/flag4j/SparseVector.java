/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.RealSparseTensorBase;
import com.flag4j.core.VectorMixin;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.operations.common.real.VectorNorms;
import com.flag4j.operations.dense.real.RealDenseElemDiv;
import com.flag4j.operations.dense.real.RealDenseOperations;
import com.flag4j.operations.dense.real.RealDenseProperties;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseVectorOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseVectorOperations;
import com.flag4j.operations.sparse.real.RealSparseVectorOperations;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseVectorOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ParameterChecks;
import com.flag4j.util.SparseDataWrapper;

import java.util.Arrays;

/**
 * Real sparse vector of arbitrary size.
 */
public class SparseVector
        extends RealSparseTensorBase<SparseVector, Vector, SparseCVector, CVector>
        implements VectorMixin<SparseVector, Vector, SparseVector, SparseCVector, Double, SparseMatrix, Matrix, SparseCMatrix> {

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
    public SparseVector(int size) {
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
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseVector(int size, int[] nonZeroEntries, int[] indices) {
        super(new Shape(size), nonZeroEntries.length,
                Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices}));
        this.size = size;
        this.indices = indices;
    }


    /**
     * Creates a sparse vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseVector(int size, double[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices})
        );

        this.size = size;
        this.indices = indices;
    }


    /**
     * Constructs a sparse vector whose non-zero values, indices, and size are specified by another sparse vector.
     * @param a Sparse vector to copy
     */
    public SparseVector(SparseVector a) {
        super(a.shape.copy(),
                a.nonZeroEntries(),
                a.entries.clone(),
                new int[a.indices.length][1]
        );
        this.indices = a.indices.clone();
        this.size = a.size;
    }


//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(int[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Integer::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(double[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Double::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(int[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Integer::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseVector(double[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = nonZeroEntries.stream().mapToDouble(Double::doubleValue).toArray();
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


    /**
     * Checks if this vector only contains zeros.
     *
     * @return True if this vector only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return entries.length==0 || ArrayUtils.isZeros(entries);
    }


    /**
     * Checks if this vectors non-zero entries only contains ones.
     *
     * @return True if this vectors non-zero entries only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return RealDenseProperties.isOnes(entries);
    }


    /**
     * Sets an index of a copy of this vector to a specified value.
     *
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
    public SparseVector set(double value, int... indices) {
        // TODO: Set should return a new instance for sparse vectors, matrices, and tensors, so entries can remain final.
        return null;
    }


    /**
     * Copies and reshapes vector. Note, since a vector is rank 1, this method simply copies the vector.
     * @param shape Shape of the new vector.
     * @return A copy of this vector.
     * @throws IllegalArgumentException If the new shape is not rank 1.
     */
    @Override
    public SparseVector reshape(Shape shape) {
        ParameterChecks.assertBroadcastable(this.shape, shape);
        ParameterChecks.assertRank(1, shape);
        return this.copy();
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
    public SparseVector reshape(int... shape) {
        return reshape(new Shape(shape));
    }


    /**
     * Flattens vector. Note, since a vector is already rank 1. This just copies the vector.
     * @return A copy of this vector.
     */
    @Override
    public SparseVector flatten() {
        return this.copy();
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public Vector join(Vector b) {
        double[] newEntries = new double[this.size + b.entries.length];

        // Copy over sparse values.
        for(int i=0; i<this.entries.length; i++) {
            newEntries[indices[i]] = entries[i];
        }

        // Copy over dense values.
        System.arraycopy(b.entries, 0, newEntries, this.size, b.entries.length);

        return new Vector(newEntries);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(CVector b) {
        CNumber[] newEntries = new CNumber[this.size + b.entries.length];

        // Copy over sparse values.
        for(int i=0; i<this.entries.length; i++) {
            newEntries[indices[i]] = new CNumber(entries[i]);
        }

        // Copy over dense values.
        ArrayUtils.arraycopy(b.entries, 0, newEntries, this.size, b.entries.length);

        return new CVector(newEntries);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public SparseVector join(SparseVector b) {
        // TODO: This must be fixed so that indices remain sorted. (Assume both vectors indices are already sorted)
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

        return new SparseVector(this.size + b.size, newEntries, newIndices);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public SparseCVector join(SparseCVector b) {
        // TODO: This must be fixed so that indices remain sorted. (Assume both vectors indices are already sorted)
        CNumber[] newEntries = new CNumber[this.entries.length + b.entries.length];
        int[] newIndices = new int[this.indices.length + b.indices.length];

        // Copy values from this vector.
        ArrayUtils.arraycopy(this.entries, 0, newEntries, 0, this.entries.length);
        // Copy values from vector b.
        ArrayUtils.arraycopy(b.entries, 0, newEntries, this.entries.length, b.entries.length);

        // Copy indices from this vector.
        System.arraycopy(this.indices, 0, newIndices, 0, this.entries.length);

        // Copy the indices from vector b with a shift.
        for(int i=0; i<b.indices.length; i++) {
            newIndices[this.indices.length+i] = b.indices[i] + this.size;
        }

        return new SparseCVector(this.size + b.size, newEntries, newIndices);
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public SparseMatrix stack(Vector b) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public SparseMatrix stack(SparseVector b) {
        // TODO: This must be fixed so that indices remain sorted. (Assume both vectors indices are already sorted)
        ParameterChecks.assertEqualShape(this.shape, b.shape);

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

        return new SparseMatrix(new Shape(2, this.size), entries, indices[0], indices[1]);
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public SparseCMatrix stack(CVector b) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public SparseCMatrix stack(SparseCVector b) {
        // TODO: This must be fixed so that indices remain sorted. (Assume both vectors indices are already sorted)
        ParameterChecks.assertEqualShape(this.shape, b.shape);

        CNumber[] entries = new CNumber[this.entries.length + b.entries.length];
        int[][] indices = new int[2][this.indices.length + b.indices.length]; // Row and column indices.

        // Copy values from this vector.
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.entries.length);
        // Copy values from vector b.
        ArrayUtils.arraycopy(b.entries, 0, entries, this.entries.length, b.entries.length);

        // Set row indices to 1 for b values (this vectors row indices are 0 which was implicitly set already).
        Arrays.fill(indices[0], this.indices.length, entries.length, 1);

        // Copy indices from this vector to the column indices.
        System.arraycopy(this.indices, 0, indices[1], 0, this.entries.length);
        // Copy indices from b vector to the column indices.
        System.arraycopy(b.indices, 0, indices[1], this.entries.length, b.entries.length);

        return new SparseCMatrix(new Shape(2, this.size), entries, indices[0], indices[1]);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public SparseMatrix stack(Vector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public SparseMatrix stack(SparseVector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public SparseCMatrix stack(CVector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public SparseCMatrix stack(SparseCVector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public Vector add(Vector B) {
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
    public SparseVector add(SparseVector B) {
        return RealSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector add(CVector B) {
        return RealComplexDenseSparseVectorOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public SparseCVector add(SparseCVector B) {
        return RealComplexSparseVectorOperations.add(B, this);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public Vector sub(Vector B) {
        return RealDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Vector add(double a) {
        return RealSparseVectorOperations.add(this, a);
    }


    /**
     * Adds specified value to all entries of this tensor.
     * If zeros are introduced, they will be explicitly stored.
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(CNumber a) {
        return RealComplexSparseVectorOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Vector to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public SparseVector sub(SparseVector B) {
        return RealSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B Vector to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector sub(CVector B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B Vector to subtract from this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public SparseCVector sub(SparseCVector B) {
        return RealComplexSparseVectorOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Vector sub(double a) {
        return RealSparseVectorOperations.sub(this, a);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVector sub(CNumber a) {
        return RealComplexSparseVectorOperations.sub(this, a);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(SparseVector B) {

    }


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(Vector B) {

    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(Vector B) {

    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(Double b) {

    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(SparseVector B) {

    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public SparseVector elemMult(Vector B) {
        return null;
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(Double b) {

    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseVector mult(double factor) {
        return new SparseVector(size, RealOperations.scalMult(entries, factor), indices.clone());
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCVector mult(CNumber factor) {
        return new SparseCVector(size, ComplexOperations.scalMult(entries, factor), indices.clone());
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public SparseVector div(double divisor) {
        return new SparseVector(size, RealOperations.scalDiv(entries, divisor), indices.clone());
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public SparseCVector div(CNumber divisor) {
        return new SparseCVector(size, ComplexOperations.scalDiv(entries, divisor), indices.clone());
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        return AggregateReal.sum(entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public SparseVector sqrt() {
        return new SparseVector(this.size, RealOperations.sqrt(entries), indices.clone());
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public SparseVector abs() {
        return new SparseVector(this.size, RealOperations.abs(entries), indices.clone());
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}. Since a vector is a rank 1 tensor, this just
     * copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseVector transpose() {
        return this.copy();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}. Since a vector is a rank 1 tensor, this just
     *      * copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseVector T() {
        return this.copy();
    }


    /**
     * Computes the reciprocals, element-wise, of this sparse vector. However, all zero entries will remain zero.
     *
     * @return A sparse vector containing the reciprocal elements of this sparse vector with zero entries preserved.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public SparseVector recip() {
        return new SparseVector(size, RealDenseOperations.recip(entries), indices.clone());
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.assertEquals(indices.length, 1);
        ParameterChecks.assertInRange(indices[0], 0, size, "index");
        // TODO: Assume indices sorted and do binary search. If index not in indices, return zero.
        return null;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public SparseVector copy() {
        return new SparseVector(this.size, this.entries.clone(), this.indices.clone());
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public SparseVector elemMult(SparseVector B) {
        return RealSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public SparseCVector elemMult(CVector B) {
        return RealComplexDenseSparseVectorOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public SparseCVector elemMult(SparseCVector B) {
        return RealComplexSparseVectorOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param b Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code b} do not have the same size.
     */
    @Override
    public SparseVector elemDiv(Vector b) {
        return RealDenseSparseVectorOperations.elemDiv(this, b);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param b Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code b} do not have the same size.
     */
    @Override
    public SparseCVector elemDiv(CVector b) {
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
    public Double inner(Vector b) {
        ParameterChecks.assertEqualShape(shape, b.shape);
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
    public Double inner(SparseVector b) {
        return RealSparseVectorOperations.inner(this, b);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector.
     */
    @Override
    public SparseVector normalize() {
        double norm = this.norm();
        return norm==0 ? new SparseVector(size) : this.div(norm);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CVector b) {
        return RealComplexDenseSparseVectorOperations.inner(b.entries, this.entries, this.indices, this.size);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(SparseCVector b) {
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
    public Matrix outer(Vector b) {
        return new Matrix(
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
    public Matrix outer(SparseVector b) {
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
    public CMatrix outer(CVector b) {
        return new CMatrix(
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
    public CMatrix outer(SparseCVector b) {
        return RealComplexSparseVectorOperations.outerProduct(this, b);
    }


    /**
     * Checks if a vector is parallel to this vector. This sparse vectors indices are assumed to be sorted lexicographically.
     * If this is not the case call {@link #sparseSort() this.sparseSort()} before calling this method.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isParallel(Vector b) {
        boolean result;

        if(this.size!=b.size) {
            result = false;
        } else if(this.size==1 || this.size==0) {
            result = true;
        } else if(this.isZeros() || b.isZeros()) {
            result = true; // Any vector is parallel to zero vector.
        } else {
            result = true;
            int sparseIndex = 0;
            double scale = this.entries[0]/b.entries[this.indices[0]];

            for(int i=0; i<b.size; i++) {
                if(i!=this.indices[sparseIndex]) {
                    // Then this index is not in the sparse vector.
                    if(b.entries[i] != 0) {
                        result = false;
                        break;
                    }

                } else {
                    if(b.entries[i]*scale != this.entries[sparseIndex]) {
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
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isPerp(Vector b) {
        boolean result;

        if(this.size!=b.size) {
            result = false;
        } else {
            result = this.inner(b)==0;
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
    public SparseMatrix toMatrix() {
        int[] rowIndices = indices.clone();
        int[] colIndices = new int[entries.length];

        return new SparseMatrix(this.size, 1, entries.clone(), rowIndices, colIndices);
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
    public SparseMatrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new SparseMatrix(this.size, 1, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new SparseMatrix(1, this.size, entries.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Finds the indices of the minimum non-zero value in this sparse vector.
     *
     * @return The indices of the minimum non-zero value in this sparse vector. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        // TODO: Implementation.
        return new int[0];
    }


    /**
     * Finds the indices of the maximum non-zero value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return new int[0];
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return VectorNorms.norm(entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    @Override
    public double norm(double p) {
        return VectorNorms.norm(entries, p);
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double infNorm() {
        return maxAbs();
    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n    The number of times to extend this vector.
     * @param axis Axis along which to extend vector.
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public SparseMatrix extend(int n, int axis) {
        return null;
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
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sparseSort() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant tensor will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public SparseCVector toComplex() {
        CNumber[] destEntries = new CNumber[entries.length];
        ArrayUtils.copy2CNumber(entries, destEntries);

        return new SparseCVector(
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
    public SparseVector flatten(int axis) {
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
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected SparseVector getSelf() {
        return this;
    }


    // TODO: Add toString method.
}


