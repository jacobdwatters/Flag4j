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

package org.flag4j.arrays_old.dense;

import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.core.VectorMixin;
import org.flag4j.core.dense_base.DenseVectorMixin;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations_old.dense.real.RealDenseEquals;
import org.flag4j.operations_old.dense.real.RealDenseVectorOperations;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseVectorOperations;
import org.flag4j.operations_old.dense_sparse.coo.real.RealDenseSparseVectorOperations;
import org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.flag4j.util.*;

import java.util.Arrays;


/**
 * Real dense vector. This class is mostly Equivalent to a real dense tensor with rank 1.
 */
@Deprecated
public class VectorOld
        extends RealDenseTensorBase<VectorOld, CVectorOld>
        implements VectorMixin<VectorOld, VectorOld, CooVectorOld, CVectorOld, Double, MatrixOld, MatrixOld, CMatrixOld>,
        DenseVectorMixin {

    /**
     * The size of this vector. That is, the number of entries in this vector.
     */
    public final int size;

    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public VectorOld(int size) {
        super(new Shape(size), new double[size]);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public VectorOld(int size, double fillValue) {
        super(new Shape(size), new double[size]);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of the specified shape filled with zeros.
     * @param shape Shape of this vector.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public VectorOld(Shape shape) {
        super(shape, new double[shape.get(0)]);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param shape Shape of the vector.
     * @param fillValue Value to fill vector with.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public VectorOld(Shape shape, double fillValue) {
        super(shape, new double[shape.get(0)]);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public VectorOld(double... entries) {
        super(new Shape(entries.length), entries.clone());
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public VectorOld(int... entries) {
        super(new Shape(entries.length), new double[entries.length]);
        this.size = shape.get(0);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a VectorOld to make copy of.
     */
    public VectorOld(VectorOld a) {
        super(a.shape, a.entries.clone());
        this.size = shape.get(0);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link VectorOld}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        VectorOld src2 = (VectorOld) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }


    /**
     * Since vectors are rank 1 tensors, this method simply copies the vector.
     *
     * @return The flattened tensor.
     */
    @Override
    public VectorOld flatten() {
        ParameterChecks.ensureBroadcastable(this.shape, shape);
        return this.copy();
    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n The number of times to extend this vector.
     * @param axis Axis along which to extend. If {@code axis=0}, then the vector will be treated as a row vector. If
     *             {@code axis=1} then the vector will be treated as a column vector.
     * @return A matrix which is the result of extending a vector {@code n} times.
     * @throws IllegalArgumentException If axis is not 0 or 1.
     */
    @Override
    public MatrixOld extend(int n, int axis) {
        MatrixOld extended;

        if(axis==0) {
            extended = new MatrixOld(n, this.size);
            for(int i=0; i<n; i++) {
                System.arraycopy(this.entries, 0, extended.entries, i*extended.numCols, this.size);
            }

        } else if(axis==1) {
            extended = new MatrixOld(this.size, n);
            double[] row = new double[n];

            for(int i=0; i<this.size; i++) {
                Arrays.fill(row, this.entries[i]);
                System.arraycopy(row, 0, extended.entries, i*extended.numCols, row.length);
            }
        } else {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, Axis2D.allAxes()));
        }

        return extended;
    }


    // TODO: Add append(double v) to append a single scalar to the end of a vector.


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public VectorOld add(CooVectorOld B) {
        return RealDenseSparseVectorOperations.add(this, B);
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
        return new CVectorOld(RealComplexDenseOperations.add(B.entries, B.shape, this.entries, this.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld add(CooCVectorOld B) {
        return RealComplexDenseSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public VectorOld sub(CooVectorOld B) {
        return RealDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld sub(CVectorOld B) {
        return new CVectorOld(RealComplexDenseOperations.sub(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld sub(CooCVectorOld B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     *
     * @param B VectorOld to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(CooVectorOld B) {
        RealDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     * @param B VectorOld to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(CooVectorOld B) {
        RealDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooVectorOld elemMult(CooVectorOld B) {
        return RealDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVectorOld elemMult(CVectorOld B) {
        return new CVectorOld(RealComplexDenseElemMult.dispatch(B.entries, B.shape, this.entries, this.shape));
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
        return RealComplexDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVectorOld elemDiv(CVectorOld B) {
        return new CVectorOld(RealComplexDenseElemDiv.dispatch(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}. For a vector, this just copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public VectorOld transpose() {
        return new VectorOld(this);
    }


    /**
     * Computes the transpose of a tensor. For a vector, this just copies the vector. <br>
     * Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public VectorOld T() {
        return new VectorOld(this);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public VectorOld join(VectorOld b) {
        VectorOld joined = new VectorOld(this.size+b.size);
        System.arraycopy(this.entries, 0, joined.entries, 0, this.size);
        System.arraycopy(b.entries, 0, joined.entries, this.size, b.size);

        return joined;
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(CVectorOld b) {
        CNumber[] entries = new CNumber[this.size+b.size];
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
        System.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CVectorOld(entries);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public VectorOld join(CooVectorOld b) {
        VectorOld joined = new VectorOld(this.size+b.size);
        System.arraycopy(this.entries, 0, joined.entries, 0, this.size);

        // Copy entries from sparse vector.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            joined.entries[this.size+index] = b.entries[i];
        }

        return joined;
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(CooCVectorOld b) {
        CVectorOld joined = new CVectorOld(this.size+b.size);
        ArrayUtils.arraycopy(this.entries, 0, joined.entries, 0, this.size);

        // Copy entries from sparse vector.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            joined.entries[this.size+index] = b.entries[i];
        }

        return joined;
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public MatrixOld stack(VectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        MatrixOld stacked = new MatrixOld(2, this.size);

        // Copy entries from each vector to the matrix.
        System.arraycopy(this.entries, 0, stacked.entries, 0, this.size);
        System.arraycopy(b.entries, 0, stacked.entries, this.size, b.size);

        return stacked;
    }


    /**
     * Stacks two vectors along columns.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public MatrixOld stack(CooVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        MatrixOld stacked = new MatrixOld(2, this.size);

        // Copy entries from dense vector to the matrix.
        System.arraycopy(this.entries, 0, stacked.entries, 0, this.size);

        // Copy entries from sparse vector to the matrix.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[stacked.numCols + index] = b.entries[i];
        }

        return stacked;
    }


    /**
     * Stacks two vectors along columns.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrixOld stack(CVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        CNumber[] entries = new CNumber[this.size+b.size];

        // Copy entries from each vector to the matrix.
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
        System.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CMatrixOld(2, this.size, entries);
    }


    /**
     * Stacks two vectors along columns.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrixOld stack(CooCVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        CMatrixOld stacked = new CMatrixOld(2, this.size);

        // Copy entries from dense vector to the matrix.
        ArrayUtils.arraycopy(this.entries, 0, stacked.entries, 0, this.size);

        // Copy entries from sparse vector to the matrix.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[stacked.numCols + index] = b.entries[i];
        }

        return stacked;
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
    public MatrixOld stack(VectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        MatrixOld stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            double[] stackedEntries = new double[2*this.size];

            int count = 0;
            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = this.entries[count];
                stackedEntries[i+1] = b.entries[count++];
            }

            stacked = new MatrixOld(this.size, 2, stackedEntries);
        }

        return stacked;
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
    public MatrixOld stack(CooVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        MatrixOld stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            double[] stackedEntries = new double[2*this.size];

            // Copy dense values.
            for(int i=0; i<this.size; i++) {
                stackedEntries[i*2] = this.entries[i];
            }

            // Copy sparse values.
            int index;
            for(int i=0; i<b.entries.length; i++) {
                index = b.indices[i];
                stackedEntries[index*2 + 1] = b.entries[i];
            }

            stacked = new MatrixOld(this.size, 2, stackedEntries);
        }

        return stacked;
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
    public CMatrixOld stack(CVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        CMatrixOld stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];

            int count = 0;

            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = new CNumber(this.entries[count]);
                stackedEntries[i+1] = b.entries[count++];
            }

            stacked = new CMatrixOld(this.size, 2, stackedEntries);
        }

        return stacked;
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
    public CMatrixOld stack(CooCVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        CMatrixOld stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];
            Arrays.fill(stackedEntries, CNumber.ZERO);

            // Copy dense values.
            for(int i=0; i<this.size; i++) {
                stackedEntries[i*2] = new CNumber(this.entries[i]);
            }

            // Copy sparse values.
            int index;
            for(int i=0; i<b.entries.length; i++) {
                index = b.indices[i];
                stackedEntries[index*2 + 1] = b.entries[i];
            }

            stacked = new CMatrixOld(this.size, 2, stackedEntries);
        }

        return stacked;
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
        return RealDenseVectorOperations.innerProduct(this.entries, b.entries);
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
        return RealDenseSparseVectorOperations.inner(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public VectorOld normalize() {
        double norm = VectorNorms.norm(this.entries);
        return norm==0 ? new VectorOld(size) : this.div(norm);
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
        return RealComplexDenseVectorOperations.innerProduct(this.entries, b.entries);
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
        return RealComplexDenseSparseVectorOperations.inner(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have exactly 3 entries.
     */
    public VectorOld cross(VectorOld b) {
        ParameterChecks.ensureEquals(3, b.size, this.size);
        double[] entries = new double[3];

        entries[0] = this.entries[1]*b.entries[2]-this.entries[2]*b.entries[1];
        entries[1] = this.entries[2]*b.entries[0]-this.entries[0]*b.entries[2];
        entries[2] = this.entries[0]*b.entries[1]-this.entries[1]*b.entries[0];

        return new VectorOld(entries);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVectorOld cross(CVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(3, b.size);
        ParameterChecks.ensureArrayLengthsEq(3, this.size);
        CNumber[] entries = new CNumber[3];

        entries[0] = b.entries[2].mult(this.entries[1]).sub(b.entries[1].mult(this.entries[2]));
        entries[1] = b.entries[0].mult(this.entries[2]).sub(b.entries[2].mult(this.entries[0]));
        entries[2] = b.entries[1].mult(this.entries[0]).sub(b.entries[0].mult(this.entries[1]));

        return new CVectorOld(entries);
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
        return RealDenseVectorOperations.dispatchOuter(this, b);
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
        return new MatrixOld(this.size, b.size,
                RealDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
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
        return new CMatrixOld(this.size, b.size,
                RealComplexDenseVectorOperations.outerProduct(this.entries, b.entries));
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
        return new CMatrixOld(this.size, b.size,
                RealComplexDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b VectorOld to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isParallel(VectorOld b) {
        // TODO: Add overloaded methods for other vector types.
        boolean result;

        if(this.size!=b.size) {
            result = false;
        } else if(this.size==1) {
            result = true;
        } else if(this.isZeros() || b.isZeros()) {
            result = true; // Any vector is parallel to zero vector.
        } else {
            result = true;
            double scale = 0;

            // Find first non-zero entry of b to compute the scaling factor.
            for(int i=0; i<b.size; i++) {
                if(b.entries[i]!=0) {
                    scale = this.entries[i]/b.entries[i];
                    break;
                }
            }

            // Ensure all entries of b are the same scalar multiple of the entries in this vector.
            for(int i=0; i<this.size; i++) {
                if(b.entries[i]*scale != this.entries[i]) {
                    result = false;
                    break;
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
        // TODO: Add overloaded methods for other vector types.
        boolean result;

        if(this.size!=b.size) {
            result = false;
        } else {
            result = this.inner(b)==0;
        }

        return result;
    }


    /**
     * Converts a vector to an equivalent matrix representing the vector as a column.
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public MatrixOld toMatrix() {
        return new MatrixOld(this.entries.length, 1, this.entries.clone());
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     *                    <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     */
    @Override
    public MatrixOld toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            return new MatrixOld(this.entries.length, 1, this.entries.clone());
        } else {
            // Convert to row vector.
            return new MatrixOld(1, this.entries.length, this.entries.clone());
        }
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor which is equivalent to this vector.
     */
    public TensorOld toTensor() {
        return new TensorOld(this.shape, this.entries.clone());
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected VectorOld makeTensor(Shape shape, double[] entries) {
        // Shape not needed to construct a dense vector.
        return new VectorOld(entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CVectorOld makeComplexTensor(Shape shape, double[] entries) {
        // Shape not needed to construct a dense vector.
        return new CVectorOld(entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CVectorOld makeComplexTensor(Shape shape, CNumber[] entries) {
        // Shape not needed to construct a dense vector.
        return new CVectorOld(entries);
    }


    /**
     * Converts this dense vector to an equivalent {@link CooVectorOld}. Note, this is likely only worthwhile for <i>very</i> sparse
     * vectors.
     * @return A {@link CooVectorOld} that is equivalent to this dense vector.
     */
    @Override
    public CooVectorOld toCoo() {
        return CooVectorOld.fromDense(this);
    }


    /**
     * Simply returns this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected VectorOld getSelf() {
        return this;
    }


    /**
     * Gets the length of a vector. Same as {@link #size()}
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return this.size;
    }


    /**
     * gets the size of this vector.
     *
     * @return The total number of entries in this vector.
     */
    @Override
    public int size() {
        return size;
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
    public MatrixOld repeat(int n, int axis) {
        ParameterChecks.ensureInRange(axis, 0, 1, "axis");
        ParameterChecks.ensureGreaterEq(0, n, "n");
        MatrixOld tiled;

        if(axis==0) {
            tiled = new MatrixOld(new Shape(n, size));

            for(int i=0; i<tiled.numRows; i++) // Set each row of the tiled matrix to be the vector values.
                System.arraycopy(entries, 0, tiled.entries, i*tiled.numCols, size);
        } else {
            tiled = new MatrixOld(new Shape(size, n));

            for(int i=0; i<tiled.numRows; i++) // Fill each row of the tiled matrix with a single value from the vector.
                Arrays.fill(tiled.entries, i*tiled.numCols, (i+1)*tiled.numCols, entries[i]);
        }

        return tiled;
    }


    /**
     * Flattens a tensor along the specified axis. For a vector, this simply copies the vector.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public VectorOld flatten(int axis) {
        return this.copy();
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of entries to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder();

        if(PrintOptions.getMaxColumns()<size) {
            // Then also get the full size of the vector.
            result.append(String.format("Full Size: %d\n", size));
        }

        result.append("[");

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

        result.append("]");

        return result.toString();
    }
}