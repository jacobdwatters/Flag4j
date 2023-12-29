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
import com.flag4j.core.VectorMixin;
import com.flag4j.core.sparse.RealSparseTensorBase;
import com.flag4j.io.PrintOptions;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.operations.common.real.VectorNorms;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseVectorOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseVectorOperations;
import com.flag4j.operations.sparse.coo.real.RealSparseEquals;
import com.flag4j.operations.sparse.coo.real.RealSparseVectorOperations;
import com.flag4j.operations.sparse.coo.real_complex.RealComplexSparseEquals;
import com.flag4j.operations.sparse.coo.real_complex.RealComplexSparseVectorOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ParameterChecks;
import com.flag4j.operations.sparse.coo.SparseDataWrapper;
import com.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Real sparse vector stored in coordinate (COO) format.
 */
public class CooVector
        extends RealSparseTensorBase<CooVector, Vector, CooCVector, CVector>
        implements VectorMixin<CooVector, Vector, CooVector, CooCVector, Double, CooMatrix, Matrix, CooCMatrix> {

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
    public CooVector(int size) {
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
    public CooVector(int size, int[] nonZeroEntries, int[] indices) {
        super(new Shape(size), nonZeroEntries.length,
                ArrayUtils.asDouble(nonZeroEntries, null),
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices})
        );
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
    public CooVector(int size, double[] nonZeroEntries, int[] indices) {
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
    public CooVector(CooVector a) {
        super(a.shape.copy(),
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
    public CooVector(int size, List<Double> entries, List<Integer> indices) {
        super(new Shape(size),
                entries.size(),
                ArrayUtils.fromDoubleList(entries),
                new int[indices.size()][1]
        );

        this.indices = ArrayUtils.fromIntegerList(indices);
        this.size = size;
    }


    /**
     * Checks if an object is equal to this vector. The object must be a vector (real, complex, dense or sparse).
     * @param b Object to compare to this vector. Valid types are {@link Vector}, {@link CooVector},
     * {@link CVector}, or {@link CooCVector}.
     * @return True if {@code b} is a vector and is element-wise equal to this vector.
     */
    @Override
    public boolean equals(Object b) {
        boolean equal = false;

        if(b instanceof CooVector) {
            CooVector vec = (CooVector) b;
            equal = RealSparseEquals.vectorEquals(this, vec);

        } else if(b instanceof Vector) {
            Vector vec = (Vector) b;
            equal = RealDenseSparseEquals.vectorEquals(vec.entries, this.entries, this.indices, this.size);

        } else if(b instanceof CooCVector) {
            CooCVector vec = (CooCVector) b;
            equal = RealComplexSparseEquals.vectorEquals(this, vec);

        } else if(b instanceof CVector) {
            CVector vec = (CVector) b;
            equal = RealComplexDenseSparseEquals.vectorEquals(vec.entries, this.entries, this.indices, this.size);
        }

        return equal;
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
    public CooVector set(double value, int... indices) {
        ParameterChecks.assertEquals(indices.length, 1);
        ParameterChecks.assertInRange(indices[0], 0, size, "index");

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

        return new CooVector(size, destEntries, destIndices);
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
        ArrayUtils.fillZeros(newEntries);

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
    public CooVector join(CooVector b) {
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

        return new CooVector(this.size + b.size, newEntries, newIndices);
    }


    /**
     * Joins specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CooCVector join(CooCVector b) {
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

        return new CooCVector(this.size + b.size, newEntries, newIndices);
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
    public CooMatrix stack(Vector b) {
        ParameterChecks.assertEqualShape(this.shape, b.shape);

        double[] destEntries = new double[nonZeroEntries + b.length()];
        int[][] indices = new int[2][nonZeroEntries + b.length()];

        // Copy sparse values and column indices (row indices will be implicitly zero)
        System.arraycopy(entries, 0, destEntries,0,  entries.length);
        System.arraycopy(this.indices, 0, indices[1], 0, this.indices.length);

        // Copy dense values. Set column indices as range and set row indices to 1.
        int[] rowIndices = new int[b.size];
        Arrays.fill(rowIndices, 1);
        System.arraycopy(b.entries, 0, destEntries, entries.length,  b.size);
        System.arraycopy(rowIndices, 0, indices[0], entries.length,  b.size);
        System.arraycopy(ArrayUtils.intRange(0, b.size), 0, indices[1], entries.length,  b.size);

        return new CooMatrix(2, b.size, destEntries, indices[0], indices[1]);
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
    public CooMatrix stack(CooVector b) {
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

        return new CooMatrix(new Shape(2, this.size), entries, indices[0], indices[1]);
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
    public CooCMatrix stack(CVector b) {
        ParameterChecks.assertEqualShape(this.shape, b.shape);

        CNumber[] destEntries = new CNumber[nonZeroEntries + b.length()];
        int[][] indices = new int[2][nonZeroEntries + b.length()];

        // Copy sparse values and column indices (row indices will be implicitly zero)
        ArrayUtils.arraycopy(entries, 0, destEntries,0,  entries.length);
        System.arraycopy(this.indices, 0, indices[1], 0, this.indices.length);

        // Copy dense values. Set column indices as range and set row indices to 1.
        int[] rowIndices = new int[b.size];
        Arrays.fill(rowIndices, 1);
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length,  b.size);
        System.arraycopy(rowIndices, 0, indices[0], entries.length,  b.size);
        System.arraycopy(ArrayUtils.intRange(0, b.size), 0, indices[1], entries.length,  b.size);

        return new CooCMatrix(2, b.size, destEntries, indices[0], indices[1]);
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
    public CooCMatrix stack(CooCVector b) {
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

        return new CooCMatrix(new Shape(2, this.size), entries, indices[0], indices[1]);
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
    public CooMatrix stack(Vector b, int axis) {
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
    public CooMatrix stack(CooVector b, int axis) {
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
    public CooCMatrix stack(CVector b, int axis) {
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
    public CooCMatrix stack(CooCVector b, int axis) {
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
    public CooVector add(CooVector B) {
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
    public CooCVector add(CooCVector B) {
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
    public CooVector sub(CooVector B) {
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
    public CooCVector sub(CooCVector B) {
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
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooVector elemMult(Vector B) {
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
    public CooVector div(double divisor) {
        return new CooVector(size, RealOperations.scalDiv(entries, divisor), indices.clone());
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CooCVector div(CNumber divisor) {
        return new CooCVector(size, ComplexOperations.scalDiv(entries, divisor), indices.clone());
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}. Since a vector is a rank 1 tensor, this just
     * copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooVector transpose() {
        return this.copy();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}. Since a vector is a rank 1 tensor, this just
     *      * copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooVector T() {
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
        ParameterChecks.assertEquals(indices.length, 1);
        ParameterChecks.assertInRange(indices[0], 0, size, "index");

        int idx = Arrays.binarySearch(this.indices, indices[0]);
        return idx>=0 ? entries[idx] : 0;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CooVector copy() {
        return new CooVector(this.size, this.entries.clone(), this.indices.clone());
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooVector elemMult(CooVector B) {
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
    public CooCVector elemMult(CVector B) {
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
    public CooCVector elemMult(CooCVector B) {
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
    public CooVector elemDiv(Vector b) {
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
    public CooCVector elemDiv(CVector b) {
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
    public Double inner(CooVector b) {
        return RealSparseVectorOperations.inner(this, b);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector.
     */
    @Override
    public CooVector normalize() {
        double norm = this.norm();
        return norm==0 ? new CooVector(size) : this.div(norm);
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
    public CNumber inner(CooCVector b) {
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
    public Matrix outer(CooVector b) {
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
    public CMatrix outer(CooCVector b) {
        return RealComplexSparseVectorOperations.outerProduct(this, b);
    }


    /**
     * Checks if a vector is parallel to this vector. This sparse vectors indices are assumed to be sorted lexicographically.
     * If this is not the case call {@link #sortIndices() this.sparseSort()} before calling this method.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isParallel(Vector b) {
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
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isPerp(Vector b) {
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
    public CooMatrix toMatrix() {
        int[] rowIndices = indices.clone();
        int[] colIndices = new int[entries.length];

        return new CooMatrix(this.size, 1, entries.clone(), rowIndices, colIndices);
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
    public CooMatrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new CooMatrix(this.size, 1, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new CooMatrix(1, this.size, entries.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor which is equivalent to this vector.
     */
    public CooTensor toTensor() {
        return new CooTensor(
                this.shape.copy(),
                this.entries.clone(),
                RealDenseTranspose.blockedIntMatrix(new int[][]{this.indices.clone()})
        );
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
     * @param n    The number of times to extend this vector. Must be a positive value.
     * @param axis Axis along which to extend. If {@code axis=0}, then the vector will be treated as a row vector. If
     *    {@code axis=1} then the vector will be treated as a column vector.
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public CooMatrix extend(int n, int axis) {
        ParameterChecks.assertGreaterEq(1, n, "n");
        ParameterChecks.assertAxis2D(axis);

        int[][] matIndices = new int[2][n*nonZeroEntries];
        double[] matEntries = new double[n*nonZeroEntries];
        Shape matShape;

        if(axis==0) {
            matShape = new Shape(n, this.size);
            int[] rowIndices = new int[indices.length];

            for(int i=0; i<n; i++) {
                Arrays.fill(rowIndices, i);
                System.arraycopy(entries, 0, matEntries, (n-1)*i, nonZeroEntries);
                System.arraycopy(rowIndices, 0, matIndices[0], (n-1)*i, nonZeroEntries);
                System.arraycopy(indices, 0, matIndices[1], (n-1)*i, nonZeroEntries);
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

        return new CooMatrix(matShape, matEntries, matIndices[0], matIndices[1]);
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
    protected CooVector makeTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooVector(size, entries, RealDenseTranspose.blockedIntMatrix(indices)[0]);
    }


    /**
     * A factory for creating a real dense tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected Vector makeDenseTensor(Shape shape, double[] entries) {
        return new Vector(entries);
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
    protected CooCVector makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CooCVector(size, entries, RealDenseTranspose.blockedIntMatrix(indices)[0]);
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
    public CooCVector toComplex() {
        CNumber[] destEntries = new CNumber[entries.length];
        ArrayUtils.copy2CNumber(entries, destEntries);

        return new CooCVector(
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
    public CooVector flatten(int axis) {
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
    protected CooVector getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooVector tensor, double relTol, double absTol) {
        return RealSparseEquals.allCloseVector(this, tensor, relTol, absTol);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public Vector toDense() {
        double[] entries = new double[size];

        for(int i=0; i<nonZeroEntries; i++) {
            entries[indices[i]] = this.entries[i];
        }

        return new Vector(entries);
    }


    /**
     * Creates a sparse tensor from a dense tensor.
     *
     * @param src Dense tensor to convert to a sparse tensor.
     * @return A sparse tensor which is equivalent to the {@code src} dense tensor.
     */
    public static CooVector fromDense(Vector src) {
        List<Double> nonZeroEntries = new ArrayList<>((int) (src.entries.length*0.8));
        List<Integer> indices = new ArrayList<>((int) (src.entries.length*0.8));

        // Fill entries with non-zero values.
        for(int i=0; i<src.entries.length; i++) {
            if(src.entries[i]!=0) {
                nonZeroEntries.add(src.entries[i]);
                indices.add(i);
            }
        }

        return new CooVector(
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
        int size = nonZeroEntries;
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


