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

package org.flag4j.arrays.dense;

import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.core.VectorMixin;
import org.flag4j.core.dense_base.DenseVectorMixin;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations.dense.real.RealDenseEquals;
import org.flag4j.operations.dense.real.RealDenseVectorOperations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseVectorOperations;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseVectorOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.flag4j.util.*;

import java.util.Arrays;


/**
 * Real dense vector. This class is mostly Equivalent to a real dense tensor with rank 1.
 */
public class Vector
        extends RealDenseTensorBase<Vector, CVector>
        implements VectorMixin<Vector, Vector, CooVector, CVector, Double, Matrix, Matrix, CMatrix>,
        DenseVectorMixin {

    /**
     * The size of this vector. That is, the number of entries in this vector.
     */
    public final int size;

    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public Vector(int size) {
        super(new Shape(size), new double[size]);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public Vector(int size, double fillValue) {
        super(new Shape(size), new double[size]);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector of the specified shape filled with zeros.
     * @param shape Shape of this vector.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public Vector(Shape shape) {
        super(shape, new double[shape.dims[0]]);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param shape Shape of the vector.
     * @param fillValue Value to fill vector with.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public Vector(Shape shape, double fillValue) {
        super(shape, new double[shape.dims[0]]);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(double... entries) {
        super(new Shape(entries.length), entries.clone());
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(int... entries) {
        super(new Shape(entries.length), new double[entries.length]);
        this.size = shape.dims[0];

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.shape.copy(), a.entries.clone());
        this.size = shape.dims[0];
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link Vector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(!(object instanceof Vector)) return false;

        Vector src2 = (Vector) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }


    /**
     * Since vectors are rank 1 tensors, this method simply copies the vector.
     *
     * @return The flattened tensor.
     */
    @Override
    public Vector flatten() {
        ParameterChecks.assertBroadcastable(this.shape, shape);
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
    public Matrix extend(int n, int axis) {
        Matrix extended;

        if(axis==0) {
            extended = new Matrix(n, this.size);
            for(int i=0; i<n; i++) {
                System.arraycopy(this.entries, 0, extended.entries, i*extended.numCols, this.size);
            }

        } else if(axis==1) {
            extended = new Matrix(this.size, n);
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
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public Vector add(CooVector B) {
        return RealDenseSparseVectorOperations.add(this, B);
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
        return new CVector(RealComplexDenseOperations.add(B.entries, B.shape, this.entries, this.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector add(CooCVector B) {
        return RealComplexDenseSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public Vector sub(CooVector B) {
        return RealDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector sub(CVector B) {
        return new CVector(RealComplexDenseOperations.sub(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector sub(CooCVector B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(CooVector B) {
        RealDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(CooVector B) {
        RealDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooVector elemMult(CooVector B) {
        return RealDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVector elemMult(CVector B) {
        return new CVector(RealComplexDenseElemMult.dispatch(B.entries, B.shape, this.entries, this.shape));
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
        return RealComplexDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVector elemDiv(CVector B) {
        return new CVector(RealComplexDenseElemDiv.dispatch(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}. For a vector, this just copies the vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Vector transpose() {
        return new Vector(this);
    }


    /**
     * Computes the transpose of a tensor. For a vector, this just copies the vector. <br>
     * Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Vector T() {
        return new Vector(this);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public Vector join(Vector b) {
        Vector joined = new Vector(this.size+b.size);
        System.arraycopy(this.entries, 0, joined.entries, 0, this.size);
        System.arraycopy(b.entries, 0, joined.entries, this.size, b.size);

        return joined;
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(CVector b) {
        CNumber[] entries = new CNumber[this.size+b.size];
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
        ArrayUtils.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CVector(entries);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public Vector join(CooVector b) {
        Vector joined = new Vector(this.size+b.size);
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
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(CooCVector b) {
        CVector joined = new CVector(this.size+b.size);
        ArrayUtils.arraycopy(this.entries, 0, joined.entries, 0, this.size);

        // Copy entries from sparse vector.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            joined.entries[this.size+index] = b.entries[i].copy();
        }

        return joined;
    }


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public Matrix stack(Vector b) {
        ParameterChecks.assertArrayLengthsEq(this.size, b.size);
        Matrix stacked = new Matrix(2, this.size);

        // Copy entries from each vector to the matrix.
        System.arraycopy(this.entries, 0, stacked.entries, 0, this.size);
        System.arraycopy(b.entries, 0, stacked.entries, this.size, b.size);

        return stacked;
    }


    /**
     * Stacks two vectors along columns.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public Matrix stack(CooVector b) {
        ParameterChecks.assertArrayLengthsEq(this.size, b.size);
        Matrix stacked = new Matrix(2, this.size);

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
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(CVector b) {
        ParameterChecks.assertArrayLengthsEq(this.size, b.size);
        CNumber[] entries = new CNumber[this.size+b.size];

        // Copy entries from each vector to the matrix.
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
        ArrayUtils.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CMatrix(2, this.size, entries);
    }


    /**
     * Stacks two vectors along columns.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(CooCVector b) {
        ParameterChecks.assertArrayLengthsEq(this.size, b.size);
        CMatrix stacked = new CMatrix(2, this.size);

        // Copy entries from dense vector to the matrix.
        ArrayUtils.arraycopy(this.entries, 0, stacked.entries, 0, this.size);

        // Copy entries from sparse vector to the matrix.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[stacked.numCols + index] = b.entries[i].copy();
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(Vector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
        Matrix stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.assertArrayLengthsEq(this.size, b.size);
            double[] stackedEntries = new double[2*this.size];

            int count = 0;
            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = this.entries[count];
                stackedEntries[i+1] = b.entries[count++];
            }

            stacked = new Matrix(this.size, 2, stackedEntries);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(CooVector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
        Matrix stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.assertArrayLengthsEq(this.size, b.size);
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

            stacked = new Matrix(this.size, 2, stackedEntries);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(CVector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
        CMatrix stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.assertArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];

            int count = 0;

            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = new CNumber(this.entries[count]);
                stackedEntries[i+1] = b.entries[count++].copy();
            }

            stacked = new CMatrix(this.size, 2, stackedEntries);
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
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(CooCVector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
        CMatrix stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.assertArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];
            ArrayUtils.fillZeros(stackedEntries);

            // Copy dense values.
            for(int i=0; i<this.size; i++) {
                stackedEntries[i*2] = new CNumber(this.entries[i]);
            }

            // Copy sparse values.
            int index;
            for(int i=0; i<b.entries.length; i++) {
                index = b.indices[i];
                stackedEntries[index*2 + 1] = b.entries[i].copy();
            }

            stacked = new CMatrix(this.size, 2, stackedEntries);
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
    public Double inner(Vector b) {
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
    public Double inner(CooVector b) {
        return RealDenseSparseVectorOperations.inner(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public Vector normalize() {
        double norm = VectorNorms.norm(this);
        return norm==0 ? new Vector(size) : this.div(norm);
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
    public CNumber inner(CooCVector b) {
        return RealComplexDenseSparseVectorOperations.inner(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have exactly 3 entries.
     */
    public Vector cross(Vector b) {
        ParameterChecks.assertEquals(3, b.size, this.size);
        double[] entries = new double[3];

        entries[0] = this.entries[1]*b.entries[2]-this.entries[2]*b.entries[1];
        entries[1] = this.entries[2]*b.entries[0]-this.entries[0]*b.entries[2];
        entries[2] = this.entries[0]*b.entries[1]-this.entries[1]*b.entries[0];

        return new Vector(entries);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVector cross(CVector b) {
        ParameterChecks.assertArrayLengthsEq(3, b.size);
        ParameterChecks.assertArrayLengthsEq(3, this.size);
        CNumber[] entries = new CNumber[3];

        entries[0] = b.entries[2].mult(this.entries[1]).sub(b.entries[1].mult(this.entries[2]));
        entries[1] = b.entries[0].mult(this.entries[2]).sub(b.entries[2].mult(this.entries[0]));
        entries[2] = b.entries[1].mult(this.entries[0]).sub(b.entries[0].mult(this.entries[1]));

        return new CVector(entries);
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
    public Matrix outer(CooVector b) {
        return new Matrix(this.size, b.size,
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
    public CMatrix outer(CVector b) {
        return new CMatrix(this.size, b.size,
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
    public CMatrix outer(CooCVector b) {
        return new CMatrix(this.size, b.size,
                RealComplexDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isParallel(Vector b) {
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
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isPerp(Vector b) {
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
    public Matrix toMatrix() {
        return new Matrix(this.entries.length, 1, this.entries.clone());
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     *                    <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     */
    @Override
    public Matrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            return new Matrix(this.entries.length, 1, this.entries.clone());
        } else {
            // Convert to row vector.
            return new Matrix(1, this.entries.length, this.entries.clone());
        }
    }


    /**
     * Converts this vector to an equivalent tensor.
     * @return A tensor which is equivalent to this vector.
     */
    public Tensor toTensor() {
        return new Tensor(this.shape.copy(), this.entries.clone());
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected Vector makeTensor(Shape shape, double[] entries) {
        // Shape not needed to construct a dense vector.
        return new Vector(entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CVector makeComplexTensor(Shape shape, double[] entries) {
        // Shape not needed to construct a dense vector.
        return new CVector(entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CVector makeComplexTensor(Shape shape, CNumber[] entries) {
        // Shape not needed to construct a dense vector.
        return new CVector(entries);
    }


    /**
     * Converts this dense vector to an equivalent {@link CooVector}. Note, this is likely only worthwhile for <i>very</i> sparse
     * vectors.
     * @return A {@link CooVector} that is equivalent to this dense vector.
     */
    @Override
    public CooVector toCoo() {
        return CooVector.fromDense(this);
    }


    /**
     * Simply returns this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected Vector getSelf() {
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
    public Matrix repeat(int n, int axis) {
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        ParameterChecks.assertGreaterEq(0, n, "n");
        Matrix tiled;

        if(axis==0) {
            tiled = new Matrix(new Shape(n, size));

            for(int i=0; i<tiled.numRows; i++) // Set each row of the tiled matrix to be the vector values.
                System.arraycopy(entries, 0, tiled.entries, i*tiled.numCols, size);
        } else {
            tiled = new Matrix(new Shape(size, n));

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
    public Vector flatten(int axis) {
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
