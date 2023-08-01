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
import com.flag4j.core.sparse.ComplexSparseTensorBase;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.complex.ComplexProperties;
import com.flag4j.operations.common.real.VectorNorms;
import com.flag4j.operations.dense.complex.AggregateDenseComplex;
import com.flag4j.operations.dense.complex.ComplexDenseOperations;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseVectorOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseVectorOperations;
import com.flag4j.operations.sparse.complex.ComplexSparseEquals;
import com.flag4j.operations.sparse.complex.ComplexSparseVectorOperations;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseEquals;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseVectorOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Complex sparse vector.
 */
public class SparseCVector
        extends ComplexSparseTensorBase<SparseCVector, CVector, SparseVector>
        implements VectorMixin<SparseCVector, CVector, SparseCVector, SparseCVector, CNumber, SparseCMatrix, CMatrix, SparseCMatrix>
{


    /**
     * The size of this vector. That is, the number of entries in this vector.
     */
    public final int size;
    /**
     * Indices of non-zero entries in this sparse vector.
     */
    public final int[] indices;


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     */
    public SparseCVector(int size) {
        super(new Shape(size), 0, new CNumber[0], new int[0][0]);
        this.size = size;
        this.indices = new int[0];
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, int[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices})
        );
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
        this.size = size;
        this.indices = indices;
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, double[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices}
        ));
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
        this.size = size;
        this.indices = indices;
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, CNumber[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices})
        );
        this.size = size;
        this.indices = indices;
    }


    /**
     * Constructs a complex sparse vector whose size, orientation, non-zero entries, and indices are specified
     * by another complex sparse vector.
     * @param a Vector to copy.
     */
    public SparseCVector(SparseCVector a) {
        super(a.shape.copy(),
                a.nonZeroEntries(),
                new CNumber[a.nonZeroEntries()],
                RealDenseTranspose.blockedIntMatrix(new int[][]{a.indices})
        );
        ArrayUtils.copy2CNumber(a.entries, super.entries);
        this.size = a.size;
        this.indices = a.indices.clone();
    }


    /**
     * Creates a sparse vector of specified size, non-zero entries, and non-zero indices.
     * @param size Full size, including zeros, of the sparse vector.
     * @param entries Non-zero entries of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public SparseCVector(int size, List<CNumber> entries, List<Integer> indices) {
        super(new Shape(size),
                entries.size(),
                entries.toArray(CNumber[]::new),
                new int[indices.size()][1]
        );

        this.indices = indices.stream().mapToInt(Integer::intValue).toArray();
        this.size = size;
    }


    /**
     * Checks if an object is equal to this vector. The object must be a vector (real, complex, dense or sparse).
     * @param b Object to compare to this vector. Valid types are {@link Vector}, {@link SparseVector},
     * {@link CVector}, or {@link SparseCVector}.
     * @return True if {@code b} is a vector and is element-wise equal to this vector.
     */
    @Override
    public boolean equals(Object b) {
        boolean equal = false;

        if(b instanceof SparseVector) {
            SparseVector vec = (SparseVector) b;
            equal = RealComplexSparseEquals.vectorEquals(vec, this);

        } else if(b instanceof Vector) {
            Vector vec = (Vector) b;
            equal = RealComplexDenseSparseEquals.vectorEquals(vec.entries, this.entries, this.indices, this.size);

        } else if(b instanceof SparseCVector) {
            SparseCVector vec = (SparseCVector) b;
            equal = ComplexSparseEquals.vectorEquals(this, vec);

        } else if(b instanceof CVector) {
            CVector vec = (CVector) b;
            equal = ComplexDenseSparseEquals.vectorEquals(vec.entries, this.entries, this.indices, this.size);
        }

        return equal;
    }


    /**
     * Creates a sparse tensor from a dense tensor.
     *
     * @param src Dense tensor to convert to a sparse tensor.
     * @return A sparse tensor which is equivalent to the {@code src} dense tensor.
     */
    public static SparseCVector fromDense(CVector src) {
        List<CNumber> nonZeroEntries = new ArrayList<>((int) (src.entries.length*0.8));
        List<Integer> indices = new ArrayList<>((int) (src.entries.length*0.8));

        // Fill entries with non-zero values.
        for(int i=0; i<src.entries.length; i++) {
            if(!src.entries[i].equals(CNumber.ZERO)) {
                nonZeroEntries.add(src.entries[i]);
                indices.add(i);
            }
        }

        return new SparseCVector(
                src.size,
                nonZeroEntries.toArray(CNumber[]::new),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected SparseCVector getSelf() {
        return this;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public SparseCVector hermTranspose() {
        return H();
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public SparseCVector H() {
        return this.conj();
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this matrix for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException  If the number of indices is not equal to the rank of this tensor.
     * @throws IndexOutOfBoundsException If any of the indices are not within this tensor.
     */
    @Override
    public SparseCVector set(CNumber value, int... indices) {
        ParameterChecks.assertEquals(indices.length, 1);
        ParameterChecks.assertInRange(indices[0], 0, size, "index");

        int idx = Arrays.binarySearch(this.indices, indices[0]);
        CNumber[] destEntries;
        int[] destIndices;

        if(idx >= 0) {
            // Then the index was found in the sparse vector.
            destIndices = this.indices.clone();
            destEntries = ArrayUtils.copyOf(entries);
            destEntries[idx] = value;

        } else{
            // Then the index was Not found int the sparse vector.
            destIndices = new int[this.indices.length+1];
            destEntries = new CNumber[this.entries.length+1];
            idx = -(idx+1);

            System.arraycopy(this.indices, 0, destIndices, 0, idx);
            destIndices[idx] = indices[0];
            System.arraycopy(this.indices, idx, destIndices, idx+1, this.indices.length-idx);

            System.arraycopy(entries, 0, destEntries, 0, idx);
            destEntries[idx] = value;
            System.arraycopy(entries, idx, destEntries, idx+1, entries.length-idx);
        }

        return new SparseCVector(size, destEntries, destIndices);
    }


    /**
     * Checks if this vector contains only real entries.
     * @return True if this vector only contains real entries. Returns false if there is at least one entry with
     * non-zero imaginary component.
     */
    @Override
    public boolean isReal() {
        return ComplexProperties.isReal(entries);
    }


    /**
     * Checks if this vector contains at least one non-real entry.
     * @return True if this vector contains at least one non-real entry. Returns false if <b>all</b> entries are real.
     */
    @Override
    public boolean isComplex() {
        return ComplexProperties.isComplex(entries);
    }


    @Override
    public SparseCVector conj() {
        return new SparseCVector(size, ComplexOperations.conj(entries), indices.clone());
    }


    @Override
    public SparseVector toReal() {
        return new SparseVector(size, ComplexOperations.toReal(entries), indices.clone());
    }


    /**
     * Converts a complex tensor to a real matrix safely. That is, first checks if the tensor only contains real values
     * and then converts to a real tensor. However, if non-real value exist, then an error is thrown.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     * @throws RuntimeException If this tensor contains at least one non-real value.
     * @see #toReal()
     */
    @Override
    public SparseVector toRealSafe() {
        return null;
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor which is equivalent to this vector.
     */
    public SparseCTensor toTensor() {
        return new SparseCTensor(
                this.shape.copy(),
                ArrayUtils.copyOf(entries),
                RealDenseTranspose.blockedIntMatrix(new int[][]{this.indices.clone()})
        );
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public SparseCVector set(double value, int... indices) {
        return set(new CNumber(value), indices);
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
    public SparseCVector reshape(Shape shape) {
        ParameterChecks.assertRank(1, shape);
        ParameterChecks.assertEquals(size, shape.get(0));
        return new SparseCVector(this);
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
    public SparseCVector reshape(int... shape) {
        ParameterChecks.assertArrayLengthsEq(1, shape.length);
        ParameterChecks.assertEquals(size, shape[0]);
        return new SparseCVector(this);
    }
    

    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public SparseCVector flatten() {
        return new SparseCVector(this);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(Vector b) {
        CNumber[] newEntries = new CNumber[this.size + b.entries.length];
        ArrayUtils.fillZeros(newEntries);

        // Copy over sparse values.
        for(int i=0; i<this.entries.length; i++) {
            newEntries[indices[i]] = entries[i];
        }

        // Copy over dense values.
        ArrayUtils.arraycopy(b.entries, 0, newEntries, this.size, b.entries.length);

        return new CVector(newEntries);
    }


    /**
     * Joints specified vector with this vector.
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
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public SparseCVector join(SparseVector b) {
        CNumber[] newEntries = new CNumber[this.entries.length + b.entries.length];
        ArrayUtils.fillZeros(newEntries);
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
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public SparseCVector join(SparseCVector b) {
        CNumber[] newEntries = new CNumber[this.entries.length + b.entries.length];
        ArrayUtils.fillZeros(newEntries);
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
    public SparseCMatrix stack(Vector b) {
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
        System.arraycopy(ArrayUtils.rangeInt(0, b.size), 0, indices[1], entries.length,  b.size);

        return new SparseCMatrix(2, b.size, destEntries, indices[0], indices[1]);
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
    public SparseCMatrix stack(SparseVector b) {
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
        System.arraycopy(ArrayUtils.rangeInt(0, b.size), 0, indices[1], entries.length,  b.size);

        return new SparseCMatrix(2, b.size, destEntries, indices[0], indices[1]);
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
    public SparseCMatrix stack(Vector b, int axis) {
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
    public SparseCMatrix stack(SparseVector b, int axis) {
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
    public CVector add(Vector B) {
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
    public SparseCVector add(SparseVector B) {
        return RealComplexSparseVectorOperations.add(this, B);
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
        return ComplexDenseSparseVectorOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public SparseCVector add(SparseCVector B) {
        return ComplexSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B Vector to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector sub(Vector B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     *
     * @param B Vector to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public SparseCVector sub(SparseVector B) {
        return RealComplexSparseVectorOperations.sub(this, B);
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
        return ComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(double a) {
        return ComplexSparseVectorOperations.add(this, a);
    }
    

    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(CNumber a) {
        return ComplexSparseVectorOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public SparseCVector sub(SparseCVector B) {
        return ComplexSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public SparseCVector elemMult(Vector B) {
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
    public SparseCVector elemMult(SparseVector B) {
        return RealComplexSparseVectorOperations.elemMult(this, B);
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
        return ComplexDenseSparseVectorOperations.elemMult(B, this);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector sub(double a) {
        return ComplexSparseVectorOperations.sub(this, a);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVector sub(CNumber a) {
        return ComplexSparseVectorOperations.sub(this, a);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected SparseCVector makeTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new SparseCVector(shape.get(0), entries, indices[0]);
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
    protected SparseVector makeRealTensor(Shape shape, double[] entries, int[][] indices) {
        return new SparseVector(shape.get(0), entries, indices[0]);
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCVector mult(double factor) {
        return new SparseCVector(
                this.size,
                ComplexOperations.scalMult(entries, factor),
                indices.clone()
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCVector mult(CNumber factor) {
        return new SparseCVector(
                this.size,
                ComplexOperations.scalMult(entries, factor),
                indices.clone()
        );
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public SparseCVector div(double divisor) {
        return new SparseCVector(
                size,
                RealComplexDenseOperations.scalDiv(entries, divisor),
                indices.clone()
        );
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
        return new SparseCVector(
                size,
                ComplexDenseOperations.scalDiv(entries, divisor),
                indices.clone()
        );
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public SparseCVector sqrt() {
        return new SparseCVector(
                size,
                ComplexOperations.sqrt(entries),
                indices.clone()
        );
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public SparseVector abs() {
        return new SparseVector(
                size,
                ComplexOperations.abs(entries),
                indices.clone()
        );
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCVector transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCVector T() {
        return new SparseCVector(this);
    }


    /**
     * Computes the reciprocals, element-wise, of this sparse vector.
     * However, all zero entries will remain zero.
     *
     * @return A sparse vector containing the reciprocal non-zero elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public SparseCVector recip() {
        return new SparseCVector(
                size,
                ComplexDenseOperations.recip(entries),
                indices.clone()
        );
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public CNumber get(int... indices) {
        return this.entries[shape.entriesIndex(indices)];
    }


    /**
     * Creates a dense copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public SparseCVector copy() {
        return new SparseCVector(this);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public SparseCVector elemMult(SparseCVector B) {
        return ComplexSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public SparseCVector elemDiv(Vector B) {
        return RealComplexDenseSparseVectorOperations.elemDiv(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public SparseCVector elemDiv(CVector B) {
        return ComplexDenseSparseVectorOperations.elemDiv(this, B);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(Vector b) {
        ParameterChecks.assertEqualShape(shape, b.shape);
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
    public CNumber inner(SparseVector b) {
        return RealComplexSparseVectorOperations.inner(this, b);
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
        return ComplexDenseSparseVectorOperations.innerProduct(this.entries, this.indices, this.size, b.entries);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public SparseCVector normalize() {
        double norm = this.norm();
        return norm==0 ? new SparseCVector(size) : this.div(norm);
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
        return ComplexSparseVectorOperations.inner(this, b);
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrix outer(Vector b) {
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
    public CMatrix outer(SparseVector b) {
        return RealComplexSparseVectorOperations.outerProduct(this, b);
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
        return new CMatrix(new Shape(this.size, b.size), ComplexDenseSparseVectorOperations.outerProduct(
                this.entries, this.indices, this.size,
                b.entries)
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
        return ComplexSparseVectorOperations.outerProduct(this, b);
    }


    /**
     * Checks if a vector is parallel to this vector.
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
            CNumber scale = new CNumber();

            // Find first non-zero entry in b and compute the scaling factor (we know there is at least one from else-if).
            for(int i=0; i<b.size; i++) {
                if(b.entries[i]!=0) {
                    scale = new CNumber(this.entries[i]).div(b.entries[this.indices[i]]);
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
                    if(this.entries[sparseIndex].sub(scale.mult(b.entries[i])).mag() > tol) {
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
            result = this.inner(b).equals(CNumber.ZERO);
        }

        return result;
    }


    /**
     * Converts a vector to an equivalent matrix.
     *
     * @return A matrix equivalent to this vector. This method will respect the orientation of the vector. That is, if
     * this vector is a row vector, then the resulting matrix will have a single row. If this vector is a column vector, then the
     * resulting matrix will have a single column.
     */
    @Override
    public SparseCMatrix toMatrix() {
        int[] rowIndices = indices.clone();
        int[] colIndices = new int[entries.length];

        return new SparseCMatrix(this.size, 1, ArrayUtils.copyOf(entries), rowIndices, colIndices);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     *                    <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     */
    @Override
    public SparseCMatrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new SparseCMatrix(this.size, 1, ArrayUtils.copyOf(entries), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new SparseCMatrix(1, this.size, ArrayUtils.copyOf(entries), rowIndices, colIndices);
        }
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        int idx = AggregateDenseComplex.argMin(entries);
        return new int[]{indices[idx]};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        int idx = AggregateDenseComplex.argMax(entries);
        return new int[]{indices[idx]};
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
     * Flattens a tensor along the specified axis. Since a vector only has 1 axis, this simply copies the vector.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public SparseCVector flatten(int axis) {
        ParameterChecks.assertInRange(axis, 0, 0, "axis");
        return new SparseCVector(this);
    }


    /**
     * Gets the full size of this vector (including non-zero entries).
     * @return The full size of this vector.
     */
    public int size() {
        return shape.totalEntries().intValueExact();
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public CVector toDense() {
        CNumber[] entries = new CNumber[size];

        for(int i=0; i<nonZeroEntries; i++) {
            entries[indices[i]] = this.entries[i];
        }

        return new CVector(entries);
    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n    The number of times to extend this vector.
     * @param axis Axis along which to extend vector.
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public SparseCMatrix extend(int n, int axis) {
        ParameterChecks.assertGreaterEq(1, n, "n");
        ParameterChecks.assertAxis2D(axis);

        int[][] matIndices = new int[2][n*nonZeroEntries];
        CNumber[] matEntries = new CNumber[n*nonZeroEntries];
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
            int[] colIndices = ArrayUtils.rangeInt(0, n);

            for(int i=0; i<entries.length; i++) {
                Arrays.fill(rowIndices, indices[i]);

                ArrayUtils.fill(matEntries, (entries.length+1)*i, (entries.length+1)*i + n, entries[i]);
                System.arraycopy(rowIndices, 0, matIndices[0], (entries.length+1)*i, n);
                System.arraycopy(colIndices, 0, matIndices[1], (entries.length+1)*i, n);
            }
        }

        return new SparseCMatrix(matShape, matEntries, matIndices[0], matIndices[1]);
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return size;
    }
}
