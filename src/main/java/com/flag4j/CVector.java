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
import com.flag4j.operations.common.complex.AggregateComplex;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.dense.complex.AggregateDenseComplex;
import com.flag4j.operations.dense.complex.ComplexDenseOperations;
import com.flag4j.operations.dense.complex.ComplexDenseProperties;
import com.flag4j.operations.dense.complex.ComplexDenseVectorOperations;
import com.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import com.flag4j.operations.dense.real_complex.RealComplexDenseVectorOperations;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseVectorOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseVectorOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.Arrays;

/**
 * Complex dense vector. This class is mostly equivalent to a rank 1 complex tensor.
 */
public class CVector extends VectorBase<CNumber[]> implements
        VectorComparisonsMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber>,
    VectorManipulationsMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber,
            CMatrix, CMatrix, SparseCMatrix, CMatrix>,
        VectorOperationsMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber,
                    CMatrix, CMatrix, SparseCMatrix, CMatrix>,
        VectorPropertiesMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber> {


    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public CVector(int size) {
        super(size, new CNumber[size]);
        ArrayUtils.fillZeros(super.entries);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, double fillValue) {
        super(size, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, CNumber fillValue) {
        super(size, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(double[] entries) {
        super(entries.length, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(int[] entries) {
        super(entries.length, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(CNumber[] entries) {
        super(entries.length, entries);
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another real vector.
     * @param a Real vector to copy.
     */
    public CVector(Vector a) {
        super(a.size(), new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another complex vector.
     * @param a Complex vector to copy.
     */
    public CVector(CVector a) {
        super(a.size(), new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
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
        return ComplexDenseProperties.isOnes(this.entries);
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
            equal = ArrayUtils.equals(vec.entries, this.entries);
        } else if(b instanceof CVector) {
            CVector vec = (CVector) b;
            equal = Arrays.equals(this.entries, vec.entries);
        } else if(b instanceof SparseVector) {
            SparseVector vec = (SparseVector) b;
            equal = RealComplexDenseSparseEquals.vectorEquals(this.entries, vec.entries, vec.indices, vec.size);
        } else if(b instanceof SparseCVector) {
            SparseCVector vec = (SparseCVector) b;
            equal = ComplexDenseSparseEquals.vectorEquals(this.entries, vec.entries, vec.indices, vec.size);
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
        this.entries[indices[0]] = new CNumber(value);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    public void set(CNumber value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(1, indices.length);
        this.entries[indices[0]] = value;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CVector add(CVector B) {
        return new CVector(ComplexDenseOperations.add(this.entries, this.shape,
                B.entries, B.shape));
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
    public CVector sub(SparseVector B) {
        return RealComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(double a) {
        return new CVector(ComplexDenseOperations.add(this.entries, a));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(CNumber a) {
        return new CVector(ComplexDenseOperations.add(this.entries, a));
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CVector sub(CVector B) {
        return new CVector(ComplexDenseOperations.sub(this.entries, this.shape,
                B.entries, B.shape));
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
        return ComplexDenseSparseVectorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(SparseVector B) {
        RealComplexDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void addEq(Vector B) {
        RealComplexDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    public void addEq(SparseCVector B) {
        ComplexDenseSparseVectorOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(Vector B) {
        RealComplexDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    @Override
    public void subEq(SparseVector B) {
        RealComplexDenseSparseVectorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     *
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    public void subEq(SparseCVector B) {
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
        return new CVector(RealComplexDenseOperations.elemMult(this.entries, this.shape, B.entries, B.shape));
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
    public CVector elemMult(CVector B) {
        return new CVector(ComplexDenseOperations.elemMult(this.entries, this.shape, B.entries, B.shape));
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
        return new CVector(RealComplexDenseOperations.elemDiv(this.entries, this.shape, B.entries, B.shape));
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
        return new CVector(ComplexDenseOperations.elemDiv(this.entries, this.shape, B.entries, B.shape));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector sub(double a) {
        return new CVector(ComplexDenseOperations.sub(this.entries, a));
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVector sub(CNumber a) {
        return new CVector(ComplexDenseOperations.sub(this.entries, a));
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(CVector B) {
        ComplexDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(CNumber b) {
        ComplexDenseOperations.addEq(this.entries, b);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(Double b) {
        RealComplexDenseOperations.addEq(this.entries, b);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(CVector B) {
        ComplexDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(CNumber b) {
        ComplexDenseOperations.subEq(this.entries, b);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(Double b) {
        RealComplexDenseOperations.subEq(this.entries, b);
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CVector scalMult(double factor) {
        return new CVector(ComplexDenseOperations.scalMult(this.entries, factor));
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CVector scalMult(CNumber factor) {
        return new CVector(ComplexDenseOperations.scalMult(this.entries, factor));
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CVector scalDiv(double divisor) {
        return new CVector(ComplexDenseOperations.scalDiv(this.entries, divisor));
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
        return new CVector(ComplexDenseOperations.scalDiv(this.entries, divisor));
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public CNumber sum() {
        return AggregateComplex.sum(this.entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public CVector sqrt() {
        return new CVector(ComplexOperations.sqrt(this.entries));
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public CVector abs() {
        return new CVector(ComplexOperations.abs(this.entries));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     * This has no effect on a vector.
     * @return The transpose of this tensor.
     */
    @Override
    public CVector transpose() {
        return T();
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
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public CVector recip() {
        return new CVector(ComplexDenseOperations.recep(this.entries));
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
        ParameterChecks.assertArrayLengthsEq(1, indices.length);
        return this.entries[indices[0]];
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CVector copy() {
        return new CVector(this);
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public CNumber min() {
        return AggregateComplex.minAbs(this.entries);
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public CNumber max() {
        return AggregateComplex.maxAbs(this.entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public CNumber minAbs() {
        return min();
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public CNumber maxAbs() {
        return max();
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return new int[]{AggregateDenseComplex.argMin(this.entries)};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return new int[]{AggregateDenseComplex.argMax(this.entries)};
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        double norm = 0;
        double mag;

        for(int i=0; i<this.size; i++) {
            mag = this.entries[i].magAsDouble();
            norm += mag*mag;
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
                return maxAbs().re; // Maximum / infinite norm.
            } else {
                return minAbs().re; // Minimum norm.
            }
        } else {
            double norm = 0;

            for(int i=0; i<this.size; i++) {
                norm += Math.pow(this.entries[i].magAsDouble(), p);
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
        return maxAbs().re;
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
            throw new IllegalArgumentException(ErrorMessages.axisErr(axis, Axis2D.allAxes()));
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
     * Joints specified vector with this vector.
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
    public CVector join(SparseVector b) {
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
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(Matrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
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
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(SparseMatrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
     */
    @Override
    public CMatrix stack(SparseVector b) {
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
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(CMatrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
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
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(SparseCMatrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
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
    public CVector add(SparseVector B) {
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
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber inner(SparseVector b) {
        return RealComplexDenseSparseVectorOperations.innerProduct(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector.
     */
    @Override
    public CVector normalize() {
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
    public CNumber inner(SparseCVector b) {
        return ComplexDenseSparseVectorOperations.innerProduct(this.entries, b.entries, b.indices, b.size);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    @Override
    public CVector cross(Vector b) {
        ParameterChecks.assertArrayLengthsEq(3, b.size);
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
    @Override
    public CVector cross(CVector b) {
        ParameterChecks.assertArrayLengthsEq(3, b.size);
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
    public CMatrix outer(SparseVector b) {
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
    public CMatrix outer(SparseCVector b) {
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

        return new CTensor(this.shape.copy(), this.entries);
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
}
