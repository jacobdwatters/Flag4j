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
import com.flag4j.core.*;
import com.flag4j.operations.common.real.Aggregate;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.operations.dense.real.*;
import com.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import com.flag4j.operations.dense.real_complex.RealComplexDenseVectorOperations;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseVectorOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseVectorOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.Arrays;


/**
 * Real dense vector. This class is mostly Equivalent to a real dense tensor with rank 1.
 */
public class Vector extends VectorBase<double[]> implements
        VectorComparisonsMixin<Vector, Vector, SparseVector, CVector, Vector, Double>,
        VectorManipulationsMixin<Vector, Vector, SparseVector, CVector, Vector, Double,
                    Matrix, Matrix, SparseMatrix, CMatrix>,
        VectorOperationsMixin<Vector, Vector, SparseVector, CVector, Vector, Double,
                    Matrix, Matrix, SparseMatrix, CMatrix>,
        VectorPropertiesMixin<Vector, Vector, SparseVector, CVector, Vector, Double> {
    

    /**
     * Creates a vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public Vector(int size) {
        super(size, new double[size]);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public Vector(int size, double fillValue) {
        super(size, new double[size]);
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Creates a vector of the specified shape filled with zeros.
     * @param shape Shape of this vector.
     * @throws IllegalArgumentException If the shapes is not rank 1.
     */
    public Vector(Shape shape) {
        super(shape, new double[shape.dims[0]]);
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
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(double[] entries) {
        super(entries.length, entries.clone());
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(int[] entries) {
        super(entries.length, new double[entries.length]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.entries.length, a.entries.clone());
    }



    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return ArrayUtils.isZeros(this.entries);
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return RealDenseProperties.isOnes(this.entries);
    }


    /**
     * Checks if two this vector is numerically element-wise equal to another object. Objects which can be equal are, {@link Vector},
     * {@link CVector}, {@link SparseVector}, {@link SparseCVector}.
     *
     * @param b Object to compare to this vector.
     * @return True if this vector and object {@code b} are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(Object b) {
        boolean equal = false;

        if(b instanceof Vector) {
            Vector vec = (Vector) b;
            equal = RealDenseEquals.vectorEquals(this.entries, vec.entries);
        } else if(b instanceof CVector) {
            CVector vec = (CVector) b;
            equal = ArrayUtils.equals(this.entries, vec.entries);
        } else if(b instanceof SparseVector) {
            SparseVector vec = (SparseVector) b;
            equal = RealDenseSparseEquals.vectorEquals(this.entries, vec.entries, vec.indices, vec.size);
        } else if(b instanceof SparseCVector) {
            SparseCVector vec = (SparseCVector) b;
            equal = RealComplexDenseSparseEquals.vectorEquals(this.entries, vec.entries, vec.indices, vec.size);
        }

        return equal;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    @Override
    public void set(double value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(1, indices.length);
        this.entries[indices[0]] = value;
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
            throw new IllegalArgumentException(ErrorMessages.axisErr(axis, Axis2D.allAxes()));
        }

        return extended;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Vector add(Vector B) {
        return new Vector(RealDenseOperations.add(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public Vector add(SparseVector B) {
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
    public CVector add(SparseCVector B) {
        return RealComplexDenseSparseVectorOperations.add(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Vector add(double a) {
        return new Vector(RealDenseVectorOperations.add(this.entries, a));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(CNumber a) {
        return new CVector(RealComplexDenseVectorOperations.add(this.entries, a));
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Vector sub(Vector B) {
        return new Vector(RealDenseOperations.sub(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public Vector sub(SparseVector B) {
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
    public CVector sub(SparseCVector B) {
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
    public void addEq(SparseVector B) {
        RealDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(Vector B) {
        RealDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(Vector B) {
        RealDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }
    

    /**
     * Computes the element-wise addition between this vector and the specified vector. The result is stored in this
     * vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(SparseVector B) {
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
    public Vector elemMult(Vector B) {
        return new Vector(RealDenseOperations.elemMult(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public SparseVector elemMult(SparseVector B) {
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
        return new CVector(RealComplexDenseOperations.elemMult(B.entries, B.shape, this.entries, this.shape));
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
    public Vector elemDiv(Vector B) {
        return new Vector(RealDenseOperations.elemDiv(this.entries, this.shape, B.entries, B.shape));
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
        return new CVector(RealComplexDenseOperations.elemDiv(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Vector sub(double a) {
        return new Vector(RealDenseOperations.sub(this.entries, a));
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVector sub(CNumber a) {
        return new CVector(RealComplexDenseOperations.sub(this.entries, a));
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(Double b) {
        RealDenseOperations.addEq(this.entries, b);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(Double b) {
        RealDenseOperations.subEq(this.entries, b);
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public Vector scalMult(double factor) {
        return new Vector(RealOperations.scalMult(this.entries, factor));
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CVector scalMult(CNumber factor) {
        return new CVector(RealComplexDenseOperations.scalMult(this.entries, factor));
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public Vector scalDiv(double divisor) {
        return new Vector(RealDenseOperations.scalDiv(this.entries, divisor));
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CVector scalDiv(CNumber divisor) {
        return new CVector(RealComplexDenseOperations.scalDiv(this.entries, divisor));
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        return Aggregate.sum(this.entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public Vector sqrt() {
        return new Vector(RealOperations.sqrt(this.entries));
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public Vector abs() {
        return new Vector(RealOperations.abs(this.entries));
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
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public Vector recip() {
        return new Vector(RealDenseOperations.recip(this.entries));
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
        ParameterChecks.assertArrayLengthsEq(1, indices.length);
        return this.entries[indices[0]];
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public Vector copy() {
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
    public Vector join(SparseVector b) {
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
    public CVector join(SparseCVector b) {
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
    public Matrix stack(SparseVector b) {
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
    public CMatrix stack(SparseCVector b) {
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
    public Double inner(SparseVector b) {
        return RealDenseSparseVectorOperations.innerProduct(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector.
     */
    @Override
    public Vector normalize() {
        return this.scalDiv(this.norm());
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
    public CNumber inner(SparseCVector b) {
        return RealComplexDenseSparseVectorOperations.innerProduct(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have exactly 3 entries.
     */
    @Override
    public Vector cross(Vector b) {
        ParameterChecks.assertArrayLengthsEq(3, b.size);
        ParameterChecks.assertArrayLengthsEq(3, this.size);
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
    @Override
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
        return new Matrix(this.size, b.size,
                RealDenseVectorOperations.outerProduct(this.entries, b.entries));
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
    public CMatrix outer(SparseCVector b) {
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
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double min() {
        return Aggregate.min(this.entries);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double max() {
        return Aggregate.max(this.entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public Double minAbs() {
        return Aggregate.minAbs(this.entries);
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public Double maxAbs() {
        return Aggregate.maxAbs(this.entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return new int[]{AggregateDenseReal.argMin(this.entries)};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return new int[]{AggregateDenseReal.argMax(this.entries)};
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        double norm = 0;

        for(int i=0; i<this.size; i++) {
            norm += this.entries[i]*this.entries[i];
        }

        return Math.sqrt(norm);
    }


    /**
     * Computes the p-norm of this tensor. Warning, if p is large in absolute value, overflow errors may occur.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.
     *          - If p is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.
     * @return The p-norm of this tensor.
     */
    @Override
    public double norm(double p) {

        if(Double.isInfinite(p)) {
            if(p > 0) {
                return maxAbs(); // Maximum / infinite norm.
            } else {
                return minAbs(); // Minimum norm.
            }
        } else {
            double norm = 0;

            for(int i=0; i<this.size; i++) {
                norm += Math.pow(Math.abs(this.entries[i]), p);
            }

            return Math.pow(norm, 1.0/p);
        }
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double infNorm() {
        return Aggregate.maxAbs(this.entries);
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
}
