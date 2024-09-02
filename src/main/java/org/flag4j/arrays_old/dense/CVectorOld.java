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
import org.flag4j.arrays.Shape;
import org.flag4j.core.VectorMixin;
import org.flag4j.core.dense_base.ComplexDenseTensorBase;
import org.flag4j.core.dense_base.DenseVectorMixin;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations_old.dense.complex.ComplexDenseEquals;
import org.flag4j.operations_old.dense.complex.ComplexDenseVectorOperations;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseVectorOperations;
import org.flag4j.operations_old.dense_sparse.coo.complex.ComplexDenseSparseVectorOperations;
import org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.flag4j.util.*;

import java.util.Arrays;

/**
 * Complex dense vector. This class is mostly equivalent to a rank 1 complex tensor.
 */
@Deprecated
public class CVectorOld extends ComplexDenseTensorBase<CVectorOld, VectorOld>
        implements VectorMixin<CVectorOld, CVectorOld, CooCVectorOld, CVectorOld, CNumber, CMatrixOld, CMatrixOld, CMatrixOld>,
        DenseVectorMixin {

    /**
     * The size of this vector. That is, the total number of entries in this vector.
     */
    public final int size;


    /**
     * Creates a column vector of specified size filled with zeros.
     *
     * @param size Size of the vector.
     */
    public CVectorOld(int size) {
        super(new Shape(size), new CNumber[size]);
        Arrays.fill(super.entries, CNumber.ZERO);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     *
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVectorOld(int size, double fillValue) {
        super(new Shape(size), new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector of specified size filled with a specified value.
     *
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVectorOld(int size, CNumber fillValue) {
        super(new Shape(size), new CNumber[size]);
        Arrays.fill(super.entries, fillValue);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     *
     * @param entries Entries for this column vector.
     */
    public CVectorOld(double... entries) {
        super(new Shape(entries.length), new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     *
     * @param entries Entries for this column vector.
     */
    public CVectorOld(int... entries) {
        super(new Shape(entries.length), new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
        this.size = shape.get(0);
    }


    /**
     * Creates a vector with specified entries.
     *
     * @param entries Entries for this column vector.
     */
    public CVectorOld(CNumber... entries) {
        super(new Shape(entries.length), entries);
        this.size = shape.get(0);
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another real vector.
     *
     * @param a Real vector to copy.
     */
    public CVectorOld(VectorOld a) {
        super(a.shape, new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
        this.size = shape.get(0);
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another complex vector.
     *
     * @param a Complex vector to copy.
     */
    public CVectorOld(CVectorOld a) {
        super(a.shape, new CNumber[a.totalEntries().intValue()]);
        System.arraycopy(a.entries, 0, super.entries, 0, a.entries.length);
        this.size = shape.get(0);
    }


    /**
     * Constructs an empty complex dense vector whose entries are {@code null}. Note, this is in contrast to {@link #CVectorOld(int)}
     * which constructs the zero vector of a specified length.
     *
     * @param size Size of the empty vector to construct.
     *
     * @return An empty dense complex vector containing null values.
     *
     * @throws NegativeArraySizeException If size is negative.
     */
    public static CVectorOld getEmpty(int size) {
        return new CVectorOld(new CNumber[size]);
    }


    /**
     * Checks if an object is equal to this vector object.
     *
     * @param object Object to check equality with this vector.
     *
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CVectorOld}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CVectorOld src2 = (CVectorOld) object;

        return ComplexDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @return The result of the element-wise vector addition.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld add(CooCVectorOld B) {
        return ComplexDenseSparseVectorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(VectorOld B) {
        RealComplexDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B VectorOld to subtract this vector.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(VectorOld B) {
        RealComplexDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @return The result of the element-wise vector addition.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld sub(VectorOld B) {
        return new CVectorOld(RealComplexDenseOperations.sub(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @return The result of the element-wise vector addition.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld sub(CooVectorOld B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @return The result of the element-wise vector addition.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld sub(CooCVectorOld B) {
        return ComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(CooVectorOld B) {
        RealComplexDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    public void addEq(CooCVectorOld B) {
        ComplexDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(CooVectorOld B) {
        RealComplexDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    public void subEq(CooCVectorOld B) {
        ComplexDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     *
     * @return The vector resulting from the element-wise multiplication.
     *
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVectorOld elemMult(VectorOld B) {
        return new CVectorOld(RealComplexDenseElemMult.dispatch(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     *
     * @return The vector resulting from the element-wise multiplication.
     *
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooCVectorOld elemMult(CooVectorOld B) {
        return RealComplexDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise multiply to this vector.
     *
     * @return The vector resulting from the element-wise multiplication.
     *
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CooCVectorOld elemMult(CooCVectorOld B) {
        return ComplexDenseSparseVectorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     *
     * @param B VectorOld to element-wise divide this vector by.
     *
     * @return The vector resulting from the element-wise division.
     *
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    @Override
    public CVectorOld elemDiv(VectorOld B) {
        return new CVectorOld(RealComplexDenseElemDiv.dispatch(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * This has no effect on a vector.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CVectorOld T() {
        return new CVectorOld(this);
    }


    /**
     * Factory to create a real tensor with the specified shape and size.
     *
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     *
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected VectorOld makeRealTensor(Shape shape, double[] entries) {
        // Shape not needed to make vector.
        return new VectorOld(entries);
    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n The number of times to extend this vector.
     * @param axis Axis along which to extend vector. If axis=0 this vector is treated as a row vector and extended along rows.
     * If axis=1 this vector is treated as a column vector and extended along the columns.
     *
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public CMatrixOld extend(int n, int axis) {
        CMatrixOld extended;

        if(axis == 0) {
            extended = new CMatrixOld(n, this.size);
            for(int i = 0; i < n; i++) {
                System.arraycopy(this.entries, 0, extended.entries, i*extended.numCols, this.size);
            }

        } else if(axis == 1) {
            extended = new CMatrixOld(this.size, n);
            CNumber[] row = new CNumber[n];

            for(int i = 0; i < this.size; i++) {
                Arrays.fill(row, this.entries[i]);
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
     * @param b VectorOld to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(VectorOld b) {
        CNumber[] entries = new CNumber[this.size + b.size];
        System.arraycopy(this.entries, 0, entries, 0, this.size);
        ArrayUtils.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CVectorOld(entries);
    }


    /**
     * Creates a new vector which is the result of joining the specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(CVectorOld b) {
        CNumber[] entries = new CNumber[this.size + b.size];
        System.arraycopy(this.entries, 0, entries, 0, this.size);
        System.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CVectorOld(entries);
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(CooVectorOld b) {
        CVectorOld joined = new CVectorOld(this.size + b.size);
        System.arraycopy(this.entries, 0, joined.entries, 0, this.size);

        // Copy entries from sparse vector.
        int index;
        for(int i = 0; i < b.entries.length; i++) {
            index = b.indices[i];
            joined.entries[this.size + index] = new CNumber(b.entries[i]);
        }

        return joined;
    }


    /**
     * Joints specified vector with this vector.
     *
     * @param b VectorOld to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CVectorOld join(CooCVectorOld b) {
        CVectorOld joined = new CVectorOld(this.size + b.size);
        System.arraycopy(this.entries, 0, joined.entries, 0, this.size);

        // Copy entries from sparse vector.
        int index;
        for(int i = 0; i < b.entries.length; i++) {
            index = b.indices[i];
            joined.entries[this.size + index] = b.entries[i];
        }

        return joined;
    }


    /**
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     *
     * @return The result of stacking this vector and vector b.
     *
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrixOld stack(VectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        CMatrixOld stacked = new CMatrixOld(2, this.size);

        // Copy entries from each vector to the matrix.
        System.arraycopy(this.entries, 0, stacked.entries, 0, this.size);
        ArrayUtils.arraycopy(b.entries, 0, stacked.entries, this.size, b.size);

        return stacked;
    }


    /**
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     *
     * @return The result of stacking this vector and vector b.
     *
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrixOld stack(CooVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        CMatrixOld stacked = new CMatrixOld(2, this.size);

        // Copy entries from dense vector to the matrix.
        System.arraycopy(this.entries, 0, stacked.entries, 0, this.size);

        // Copy entries from sparse vector to the matrix.
        int index;
        for(int i = 0; i < b.entries.length; i++) {
            index = b.indices[i];
            stacked.entries[stacked.numCols + index] = new CNumber(b.entries[i]);
        }

        return stacked;
    }


    /**
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     *
     * @return The result of stacking this vector and vector b.
     *
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrixOld stack(CVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        CNumber[] entries = new CNumber[this.size + b.size];

        // Copy entries from each vector to the matrix.
        System.arraycopy(this.entries, 0, entries, 0, this.size);
        System.arraycopy(b.entries, 0, entries, this.size, b.size);

        return new CMatrixOld(2, this.size, entries);
    }


    /**
     * Stacks two vectors along columns as if they are column vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     *
     * @return The result of stacking this vector and vector b.
     *
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrixOld stack(CooCVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
        CMatrixOld stacked = new CMatrixOld(2, this.size);

        // Copy entries from dense vector to the matrix.
        System.arraycopy(this.entries, 0, stacked.entries, 0, this.size);

        // Copy entries from sparse vector to the matrix.
        int index;
        for(int i = 0; i < b.entries.length; i++) {
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
     * @param b VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrixOld stack(VectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        CMatrixOld stacked;

        if(axis == 0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];

            int count = 0;

            for(int i = 0; i < stackedEntries.length; i += 2) {
                stackedEntries[i] = this.entries[count];
                stackedEntries[i + 1] = new CNumber(b.entries[count++]);
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
     * @param b VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrixOld stack(CooVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        CMatrixOld stacked;

        if(axis == 0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];
            Arrays.fill(stackedEntries, CNumber.ZERO);

            // Copy dense values.
            for(int i = 0; i < this.size; i++) {
                stackedEntries[i*2] = this.entries[i];
            }

            // Copy sparse values.
            int index;
            for(int i = 0; i < b.entries.length; i++) {
                index = b.indices[i];
                stackedEntries[index*2 + 1] = new CNumber(b.entries[i]);
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
     * @param b VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrixOld stack(CVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        CMatrixOld stacked;

        if(axis == 0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];

            int count = 0;

            for(int i = 0; i < stackedEntries.length; i += 2) {
                stackedEntries[i] = this.entries[count];
                stackedEntries[i + 1] = b.entries[count++];
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
     * @param b VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrixOld stack(CooCVectorOld b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        CMatrixOld stacked;

        if(axis == 0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            CNumber[] stackedEntries = new CNumber[2*this.size];
            Arrays.fill(stackedEntries, CNumber.ZERO);

            // Copy dense values.
            for(int i = 0; i < this.size; i++) {
                stackedEntries[i*2] = this.entries[i];
            }

            // Copy sparse values.
            int index;
            for(int i = 0; i < b.entries.length; i++) {
                index = b.indices[i];
                stackedEntries[index*2 + 1] = b.entries[i];
            }

            stacked = new CMatrixOld(this.size, 2, stackedEntries);
        }

        return stacked;
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @return The result of the element-wise vector addition.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld add(VectorOld B) {
        return new CVectorOld(RealComplexDenseOperations.add(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B VectorOld to add to this vector.
     *
     * @return The result of the element-wise vector addition.
     *
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public CVectorOld add(CooVectorOld B) {
        return new CVectorOld(RealComplexDenseSparseVectorOperations.add(this, B));
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector b.
     *
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(VectorOld b) {
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
     *
     * @return The inner product between this vector and the vector b.
     *
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CooVectorOld b) {
        return RealComplexDenseSparseVectorOperations.inner(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public CVectorOld normalize() {
        double norm = VectorNorms.norm(this);
        return norm == 0 ? new CVectorOld(size) : this.div(norm);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector b.
     *
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CVectorOld b) {
        return ComplexDenseVectorOperations.innerProduct(this.entries, b.entries);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector b.
     *
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(CooCVectorOld b) {
        return ComplexDenseSparseVectorOperations.innerProduct(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     *
     * @return The result of the vector cross product between this vector and b.
     *
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVectorOld cross(VectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(3, b.size);
        ParameterChecks.ensureArrayLengthsEq(3, this.size);
        CNumber[] entries = new CNumber[3];

        entries[0] = this.entries[1].mult(b.entries[2]).sub(this.entries[2].mult(b.entries[1]));
        entries[1] = this.entries[2].mult(b.entries[0]).sub(this.entries[0].mult(b.entries[2]));
        entries[2] = this.entries[0].mult(b.entries[1]).sub(this.entries[1].mult(b.entries[0]));

        return new CVectorOld(entries);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     *
     * @return The result of the vector cross product between this vector and b.
     *
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    public CVectorOld cross(CVectorOld b) {
        ParameterChecks.ensureArrayLengthsEq(3, b.size);
        ParameterChecks.ensureArrayLengthsEq(3, this.size);
        CNumber[] entries = new CNumber[3];

        entries[0] = this.entries[1].mult(b.entries[2]).sub(this.entries[2].mult(b.entries[1]));
        entries[1] = this.entries[2].mult(b.entries[0]).sub(this.entries[0].mult(b.entries[2]));
        entries[2] = this.entries[0].mult(b.entries[1]).sub(this.entries[1].mult(b.entries[0]));

        return new CVectorOld(entries);
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and b.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrixOld outer(VectorOld b) {
        return new CMatrixOld(this.size, b.size,
                RealComplexDenseVectorOperations.outerProduct(this.entries, b.entries));
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and b.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrixOld outer(CooVectorOld b) {
        return new CMatrixOld(this.size, b.size,
                RealComplexDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and b.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrixOld outer(CVectorOld b) {
        return new CMatrixOld(this.size, b.size,
                ComplexDenseVectorOperations.outerProduct(this.entries, b.entries));
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and b.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrixOld outer(CooCVectorOld b) {
        return new CMatrixOld(this.size, b.size,
                ComplexDenseSparseVectorOperations.outerProduct(this.entries, b.entries, b.indices, b.size));
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b VectorOld to compare to this vector.
     *
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isParallel(VectorOld b) {
        boolean result;

        if(this.size != b.size) {
            result = false;
        } else if(this.size == 1) {
            result = true;
        } else if(this.isZeros() || b.isZeros()) {
            result = true; // Any vector is parallel to zero vector.
        } else {
            result = true;
            CNumber scale = CNumber.ZERO;

            // Find first non-zero entry of b to compute the scaling factor.
            for(int i = 0; i < b.size; i++) {
                if(b.entries[i] != 0) {
                    scale = this.entries[i].div(b.entries[i]);
                    break;
                }
            }

            // Ensure all entries of b are the same scalar multiple of the entries in this vector.
            for(int i = 0; i < this.size; i++) {
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
     * @param b VectorOld to compare to this vector.
     *
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    @Override
    public boolean isPerp(VectorOld b) {
        boolean result;

        if(this.size != b.size) result = false;
        else result = this.inner(b).equals(0);

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
    public CMatrixOld toMatrix() {
        CNumber[] entries = new CNumber[this.size];
        System.arraycopy(this.entries, 0, entries, 0, this.size);
        return new CMatrixOld(this.entries.length, 1, entries);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     * <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     * <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public CMatrixOld toMatrix(boolean columVector) {
        if(columVector) {
            return toMatrix();
        } else {
            // Convert to row vector.
            CNumber[] entries = new CNumber[this.size];
            System.arraycopy(this.entries, 0, entries, 0, this.size);
            return new CMatrixOld(1, this.entries.length, entries);
        }
    }


    /**
     * Creates a rank 1 tensor which is equivalent to this vector.
     *
     * @return A rank 1 tensor equivalent to this vector.
     */
    public CTensorOld toTensor() {
        CNumber[] entries = new CNumber[this.size];
        System.arraycopy(this.entries, 0, entries, 0, this.size);

        return new CTensorOld(this.shape, entries);
    }


    /**
     * Converts this dense vector to an equivalent {@link CooCVectorOld}. Note, this is likely only worthwhile for <i>very</i> sparse
     * vectors.
     *
     * @return A {@link CooCVectorOld} that is equivalent to this dense vector.
     */
    @Override
    public CooCVectorOld toCoo() {
        return CooCVectorOld.fromDense(this);
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
    public CVectorOld H() {
        return conj();
    }


    /**
     * Since vectors are rank 1 tensors, this method simply copies the vector.
     *
     * @param shape Shape of the new tensor.
     *
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     *
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public CVectorOld reshape(Shape shape) {
        ParameterChecks.ensureBroadcastable(this.shape, shape);
        ParameterChecks.ensureRank(shape, 1);
        return this.copy();
    }


    /**
     * Since vectors are rank 1 tensors, this method simply copies the vector.
     *
     * @return The flattened tensor.
     */
    @Override
    public CVectorOld flatten() {
        return this.copy();
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CVectorOld flatten(int axis) {
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
    public CMatrixOld repeat(int n, int axis) {
        ParameterChecks.ensureInRange(axis, 0, 1, "axis");
        ParameterChecks.ensureGreaterEq(0, n, "n");
        Shape tiledShape;
        CNumber[] tiledEntries = new CNumber[size*n];

        if(axis == 0) {
            tiledShape = new Shape(n, size);
            for(int i = 0; i < n; i++) // Set each row of the tiled matrix to be the vector values.
                System.arraycopy(entries, 0, tiledEntries, i*size, size);
        } else {
            tiledShape = new Shape(size, n);
            for(int i = 0; i < size; i++) // Fill each row of the tiled matrix with a single value from the vector.
                ArrayUtils.fill(tiledEntries, i*n, (i + 1)*n, entries[i]);
        }

        return new CMatrixOld(tiledShape, tiledEntries);
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     *
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CVectorOld makeTensor(Shape shape, CNumber[] entries) {
        return new CVectorOld(entries);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CVectorOld getSelf() {
        return this;
    }


    /**
     * Generates a human-readable string representation of this vector.
     *
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder();

        if(PrintOptions.getMaxColumns() < size) {
            // Then also get the full size of the vector.
            result.append(String.format("Full Size: %d\n", size));
        }

        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns() - 1, size - 1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i = 0; i < stopIndex; i++) {
            value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size - 1) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry.
        value = StringUtils.ValueOfRound(entries[size - 1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
