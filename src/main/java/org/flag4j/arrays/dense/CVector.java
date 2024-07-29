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
import org.flag4j.core.dense_base.ComplexDenseTensorBase;
import org.flag4j.core.dense_base.DenseVectorMixin;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations.dense.complex.ComplexDenseEquals;
import org.flag4j.operations.dense.complex.ComplexDenseVectorOperations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseVectorOperations;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseVectorOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.flag4j.util.*;

/**
 * Complex dense vector. This class is mostly equivalent to a rank 1 complex tensor.
 */
public class CVector extends ComplexDenseTensorBase<CVector, Vector>
        implements VectorMixin<CVector, CVector, CooCVector, CVector, CNumber, CMatrix, CMatrix, CMatrix>,
        DenseVectorMixin {

    /**
     * The size of this vector. That is, the total number of entries in this vector.
     */
    public final int size;

    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public CVector(int size) {
        super(new Shape(size), new CNumber[size]);
        ArrayUtils.fillZeros(super.entries);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, double fillValue) {
        super(new Shape(size), new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, CNumber fillValue) {
        super(new Shape(size), new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(double... entries) {
        super(new Shape(entries.length), new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(int... entries) {
        super(new Shape(entries.length), new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
        this.size = shape.dims[0];
    }


    /**
     * Creates a vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(CNumber... entries) {
        super(new Shape(entries.length), entries);
        this.size = shape.dims[0];
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another real vector.
     * @param a Real vector to copy.
     */
    public CVector(Vector a) {
        super(a.shape.copy(), new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
        this.size = shape.dims[0];
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another complex vector.
     * @param a Complex vector to copy.
     */
    public CVector(CVector a) {
        super(a.shape.copy(), new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
        this.size = shape.dims[0];
    }


    /**
     * Constructs an empty complex dense vector whose entries are {@code null}. Note, this is in contrast to {@link #CVector(int)}
     * which constructs the zero vector of a specified length.
     * @param size Size of the empty vector to construct.
     * @return An empty dense complex vector containing null values.
     * @throws NegativeArraySizeException If size is negative.
     */
    public static CVector getEmpty(int size) {
        return new CVector(new CNumber[size]);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CVector src2 = (CVector) object;

        return ComplexDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
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
        return ComplexDenseSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector sub(Vector B) {
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
    public CVector sub(CooVector B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
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
        return ComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(CooVector B) {
        RealComplexDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    public void addEq(CooCVector B) {
        ComplexDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(CooVector B) {
        RealComplexDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    public void subEq(CooCVector B) {
        ComplexDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVector elemMult(Vector B) {
        return new CVector(RealComplexDenseElemMult.dispatch(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooCVector elemMult(CooVector B) {
        return RealComplexDenseSparseVectorOperations.elemMult(this, B);
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
        return ComplexDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVector elemDiv(Vector B) {
        return new CVector(RealComplexDenseElemDiv.dispatch(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * This has no effect on a vector.
     * @return The transpose of this tensor.
     */
    @Override
    public CVector T() {
        return new CVector(this);
    }


    /**
     * Factory to create a real tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected Vector makeRealTensor(Shape shape, double[] entries) {
        // Shape not needed to make vector.
        return new Vector(entries);
    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n The number of times to extend this vector.
     * @param axis Axis along which to extend vector. If axis=0 this vector is treated as a row vector and extended along rows.
     *             If axis=1 this vector is treated as a column vector and extended along the columns.
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public CMatrix extend(int n, int axis) {
        CMatrix extended;

        if(axis==0) {
            extended = new CMatrix(n, this.size);
            for(int i=0; i<n; i++) {
                ArrayUtils.arraycopy(this.entries, 0, extended.entries, i*extended.numCols, this.size);
            }

        } else if(axis==1) {
            extended = new CMatrix(this.size, n);
            CNumber[] row = new CNumber[n];

            for(int i=0; i<this.size; i++) {
                ArrayUtils.fill(row, this.entries[i]);
                System.arraycopy(row, 0, extended.entries, i*extended.numCols, row.length);
            }
        } else {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, Axis2D.allAxes()));
        }

        return extended;
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(Vector b) {
        CNumber[] entries = new CNumber[this.size+b.size];
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
        ArrayUtils.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CVector(entries);
    }


    /**
     * Creates a new vector which is the result of joining the specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(CVector b) {
        CNumber[] entries = new CNumber[this.size+b.size];
        System.arraycopy(this.entries, 0, entries, 0, this.size);
        System.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CVector(entries);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVector join(CooVector b) {
        CVector joined = new CVector(this.size+b.size);
        ArrayUtils.arraycopy(this.entries, 0, joined.entries, 0, this.size);

        // Copy entries from sparse vector.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            joined.entries[this.size+index] = new CNumber(b.entries[i]);
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
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(Vector b) {
        ParameterChecks.assertArrayLengthsEq(this.size, b.size);
        CMatrix stacked = new CMatrix(2, this.size);

        // Copy entries from each vector to the matrix.
        ArrayUtils.arraycopy(this.entries, 0, stacked.entries, 0, this.size);
        ArrayUtils.arraycopy(b.entries, 0, stacked.entries, this.size, b.size);

        return stacked;
    }


    /**
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(CooVector b) {
        ParameterChecks.assertArrayLengthsEq(this.size, b.size);
        CMatrix stacked = new CMatrix(2, this.size);

        // Copy entries from dense vector to the matrix.
        ArrayUtils.arraycopy(this.entries, 0, stacked.entries, 0, this.size);

        // Copy entries from sparse vector to the matrix.
        int index;
        for(int i=0; i<b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[stacked.numCols + index] = new CNumber(b.entries[i]);
        }

        return stacked;
    }


    /**
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
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
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
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
    public CMatrix stack(Vector b, int axis) {
        ParameterChecks.assertAxis2D(axis);
        CMatrix stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.assertArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];

            int count = 0;

            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = this.entries[count].copy();
                stackedEntries[i+1] = new CNumber(b.entries[count++]);
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
    public CMatrix stack(CooVector b, int axis) {
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
                stackedEntries[i*2] = this.entries[i].copy();
            }

            // Copy sparse values.
            int index;
            for(int i=0; i<b.entries.length; i++) {
                index = b.indices[i];
                stackedEntries[index*2 + 1] = new CNumber(b.entries[i]);
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
                stackedEntries[i] = this.entries[count].copy();
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
                stackedEntries[i*2] = this.entries[i].copy();
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
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector add(Vector B) {
        return new CVector(RealComplexDenseOperations.add(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVector add(CooVector B) {
        return new CVector(RealComplexDenseSparseVectorOperations.add(this, B));
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
        return RealComplexDenseVectorOperations.innerProduct(this.entries, b.entries);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @return The inner product between this vector and the vector b.
     */
    public double innerSelf() {
        double inner = 0;

        for(CNumber value : entries) {
            inner += (value.re*value.re + value.im*value.im);
        }

        return inner;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CooVector b) {
        return RealComplexDenseSparseVectorOperations.inner(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public CVector normalize() {
        double norm = VectorNorms.norm(this);
        return norm==0 ? new CVector(size) : this.div(norm);
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
        return ComplexDenseVectorOperations.innerProduct(this.entries, b.entries);
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
        return ComplexDenseSparseVectorOperations.innerProduct(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVector cross(Vector b) {
        ParameterChecks.assertArrayLengthsEq(3, b.size);
        ParameterChecks.assertArrayLengthsEq(3, this.size);
        CNumber[] entries = new CNumber[3];

        entries[0] = this.entries[1].mult(b.entries[2]).sub(this.entries[2].mult(b.entries[1]));
        entries[1] = this.entries[2].mult(b.entries[0]).sub(this.entries[0].mult(b.entries[2]));
        entries[2] = this.entries[0].mult(b.entries[1]).sub(this.entries[1].mult(b.entries[0]));

        return new CVector(entries);
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

        entries[0] = this.entries[1].mult(b.entries[2]).sub(this.entries[2].mult(b.entries[1]));
        entries[1] = this.entries[2].mult(b.entries[0]).sub(this.entries[0].mult(b.entries[2]));
        entries[2] = this.entries[0].mult(b.entries[1]).sub(this.entries[1].mult(b.entries[0]));

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
    public CMatrix outer(Vector b) {
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
    public CMatrix outer(CooVector b) {
        return new CMatrix(this.size, b.size,
                RealComplexDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
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
                ComplexDenseVectorOperations.outerProduct(this.entries, b.entries));
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
                ComplexDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
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
            CNumber scale = new CNumber();

            // Find first non-zero entry of b to compute the scaling factor.
            for(int i=0; i<b.size; i++) {
                if(b.entries[i]!=0) {
                    scale = this.entries[i].div(b.entries[i]);
                    break;
                }
            }

            // Ensure all entries of b are the same scalar multiple of the entries in this vector.
            for(int i=0; i<this.size; i++) {
                if(!scale.mult(b.entries[i]).equals(this.entries[i])) {
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
            result = this.inner(b).equals(0);
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
    public CMatrix toMatrix() {
        CNumber[] entries = new CNumber[this.size];
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
        return new CMatrix(this.entries.length, 1, entries);
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
    public CMatrix toMatrix(boolean columVector) {
        if(columVector) {
            return toMatrix();
        } else {
            // Convert to row vector.
            CNumber[] entries = new CNumber[this.size];
            ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);
            return new CMatrix(1, this.entries.length, entries);
        }
    }


    /**
     * Creates a rank 1 tensor which is equivalent to this vector.
     * @return A rank 1 tensor equivalent to this vector.
     */
    public CTensor toTensor() {
        CNumber[] entries = new CNumber[this.size];
        ArrayUtils.arraycopy(this.entries, 0, entries, 0, this.size);

        return new CTensor(this.shape.copy(), entries);
    }


    /**
     * Converts this dense vector to an equivalent {@link CooCVector}. Note, this is likely only worthwhile for <i>very</i> sparse
     * vectors.
     * @return A {@link CooCVector} that is equivalent to this dense vector.
     */
    @Override
    public CooCVector toCoo() {
        return CooCVector.fromDense(this);
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
     * Computes the conjugate transpose of this vector. Since a vector is a rank 1 tensor, this simply
     * computes the complex conjugate of this vector.
     *
     * @return The complex conjugate of this vector.
     */
    @Override
    public CVector H() {
        return conj();
    }


    /**
     * Since vectors are rank 1 tensors, this method simply copies the vector.
     *
     * @param shape Shape of the new tensor.
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public CVector reshape(Shape shape) {
        ParameterChecks.assertBroadcastable(this.shape, shape);
        ParameterChecks.assertRank(1, shape);
        return this.copy();
    }


    /**
     * Since vectors are rank 1 tensors, this method simply copies the vector.
     *
     * @return The flattened tensor.
     */
    @Override
    public CVector flatten() {
        return this.copy();
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CVector flatten(int axis) {
        return this.copy();
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
    public CMatrix repeat(int n, int axis) {
        ParameterChecks.assertInRange(axis, 0, 1, "axis");
        ParameterChecks.assertGreaterEq(0, n, "n");
        Shape tiledShape;
        CNumber[] tiledEntries = new CNumber[size*n];

        if(axis==0) {
            tiledShape = new Shape(n, size);
            for(int i=0; i<n; i++) // Set each row of the tiled matrix to be the vector values.
                ArrayUtils.arraycopy(entries, 0, tiledEntries, i*size, size);
        } else {
            tiledShape = new Shape(size, n);
            for(int i=0; i<size; i++) // Fill each row of the tiled matrix with a single value from the vector.
                ArrayUtils.fill(tiledEntries, i*n, (i+1)*n, entries[i]);
        }

        return new CMatrix(tiledShape, tiledEntries);
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CVector makeTensor(Shape shape, CNumber[] entries) {
        return new CVector(entries);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CVector getSelf() {
        return this;
    }


    /**
     * Generates a human-readable string representation of this vector.
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

        // Get last entry.
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
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
        RealComplexDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to subtract this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(Vector B) {
        RealComplexDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }
}
