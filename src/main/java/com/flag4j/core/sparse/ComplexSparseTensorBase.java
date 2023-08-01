/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.core.sparse;

import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.ComplexTensorMixin;
import com.flag4j.operations.common.complex.AggregateComplex;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.complex.ComplexProperties;
import com.flag4j.operations.dense.complex.AggregateDenseComplex;
import com.flag4j.operations.dense.complex.ComplexDenseOperations;
import com.flag4j.operations.dense.complex.ComplexDenseProperties;
import com.flag4j.operations.dense.real.AggregateDenseReal;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;
import com.flag4j.util.SparseDataWrapper;

import java.math.BigInteger;

/**
 * This abstract class is the base class of all complex sparse tensors.
 * @param <T> Type of this tensor.
 * @param <U> Dense Tensor type.
 * @param <Y> Real tensor type.
 */
public abstract class ComplexSparseTensorBase<T, U, Y>
        extends SparseTensorBase<T, U, T, U, Y, CNumber[], CNumber>
        implements ComplexTensorMixin<T, Y> {

    /**
     * Creates a sparse tensor with specified shape.
     * @param shape Shape of this tensor.
     * @param nonZeroEntries Number of non-zero entries in the sparse tensor.
     * @param entries Non-zero entries of this sparse tensor.
     * @param indices Indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of rows in the indices array is not equal to the number of
     * elements in the entries array.
     * @throws IllegalArgumentException If the number of columns in the entries array is not equal to the rank of this
     * tensor.
     */
    public ComplexSparseTensorBase(Shape shape, int nonZeroEntries, CNumber[] entries, int[][] indices) {
        super(shape, nonZeroEntries, entries, indices);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
        ParameterChecks.assertArrayLengthsEq(nonZeroEntries, indices.length);
    }


    /**
     * Creates a deep copy of the indices of this sparse tensor.
     * @return A deep copy of the indices of this sparse tensor.
     */
    private int[][] copyIndices() {
        int[][] newIndices = new int[indices.length][indices[0].length];
        ArrayUtils.deepCopy(indices, newIndices);
        return newIndices;
    }


    /**
     * A factory for creating a complex sparse tensor.
     * @param shape Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    protected abstract T makeTensor(Shape shape, CNumber[] entries, int[][] indices);


    /**
     * A factory for creating a real sparse tensor.
     * @param shape Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    protected abstract Y makeRealTensor(Shape shape, double[] entries, int[][] indices);


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sparseSort() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Gets the number of non-zero entries stored in this sparse tensor.
     * @return The number of non-zero entries stored in this tensor.
     */
    @Override
    public int nonZeroEntries() {
        return entries.length;
    }


    @Override
    public double min() {
        return minAbs();
    }


    @Override
    public double max() {
        return maxAbs();
    }


    @Override
    public double minAbs() {
        return AggregateComplex.minAbs(entries);
    }


    @Override
    public double maxAbs() {
        return AggregateComplex.maxAbs(entries);
    }


    @Override
    public double infNorm() {
        return AggregateComplex.maxAbs(entries);
    }


    @Override
    public boolean isZeros() {
        return ArrayUtils.isZeros(entries);
    }


    /**
     * Checks if this sparse tensors non-zero values are all ones.
     * @return True if this sparse tensors non-zero values are all ones. False otherwise.
     */
    @Override
    public boolean isOnes() {
        return ComplexDenseProperties.isOnes(entries);
    }


    @Override
    public CNumber sum() {
        return AggregateComplex.sum(entries);
    }


    @Override
    public T mult(double a) {
        return makeTensor(shape.copy(), ComplexOperations.scalMult(entries, a), copyIndices());
    }


    @Override
    public T mult(CNumber a) {
        return makeTensor(shape.copy(), ComplexOperations.scalMult(entries, a), copyIndices());
    }


    @Override
    public T div(double divisor) {
        return makeTensor(shape.copy(), ComplexOperations.scalMult(entries, divisor), copyIndices());
    }


    @Override
    public T div(CNumber divisor) {
        return makeTensor(shape.copy(), ComplexOperations.scalMult(entries, divisor), copyIndices());
    }


    @Override
    public T sqrt() {
        return makeTensor(shape.copy(), ComplexOperations.sqrt(entries), copyIndices());
    }


    @Override
    public Y abs() {
        return makeRealTensor(shape.copy(), ComplexOperations.abs(entries), copyIndices());
    }


    @Override
    public T conj() {
        return makeTensor(shape.copy(), ComplexOperations.conj(entries), copyIndices());
    }


    @Override
    public T recip() {
        return makeTensor(shape.copy(), ComplexDenseOperations.recip(entries), copyIndices());
    }


    @Override
    public boolean isComplex() {
        return ComplexProperties.isComplex(entries);
    }


    @Override
    public boolean isReal() {
        return ComplexProperties.isReal(entries);
    }


    @Override
    public T round() {
        return makeTensor(shape.copy(), ComplexOperations.round(this.entries), copyIndices());
    }


    @Override
    public T round(int precision) {
        return makeTensor(shape.copy(), ComplexOperations.round(this.entries, precision), copyIndices());
    }


    @Override
    public int[] argMax() {
        int idx = AggregateDenseComplex.argMax(entries);
        return indices[idx].clone();
    }


    @Override
    public int[] argMin() {
        int idx = AggregateDenseComplex.argMin(entries);
        return indices[idx].clone();
    }


    /**
     * Converts a complex tensor to a real matrix. The imaginary component of any complex value will be ignored.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     * @see #toRealSafe()
     */
    @Override
    public Y toReal() {
        return makeRealTensor(this.shape.copy(), ComplexOperations.toReal(this.entries), copyIndices());
    }


    /**
     * Converts a complex tensor to a real matrix safely. That is, first checks if the tensor only contains real values
     * and then converts to a real tensor. However, if non-real value exist, then an error is thrown.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     * @see #toReal()
     * @throws RuntimeException If this tensor contains at least one non-real value.
     */
    @Override
    public Y toRealSafe() {
        if(!this.isReal()) {
            throw new RuntimeException("Could not safely convert from complex to real as non-real " +
                    "values exist in tensor.");
        }

        return toReal();
    }


    /**
     * Copies and reshapes tensor if possible. The total number of entries in this tensor must match the total number of entries
     * in the reshaped tensor.
     *
     * @param newShape Shape of the new tensor.
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public T reshape(Shape newShape) {
        ParameterChecks.assertBroadcastable(shape, newShape);
        return makeTensor(newShape, ArrayUtils.copyOf(entries), copyIndices());
    }


    /**
     * Copies and reshapes tensor if possible. The total number of entries in this tensor must match the total number of entries
     * in the reshaped tensor.
     *
     * @param dims Shape of the new tensor.
     * @return A tensor which is equivalent to this tensor but with the specified shape.
     * @throws IllegalArgumentException If this tensor cannot be reshaped to the specified dimensions.
     */
    @Override
    public T reshape(int... dims) {
        return reshape(new Shape(dims));
    }


    @Override
    public T flatten() {
        return makeTensor(
                new Shape(shape.totalEntries().intValueExact()),
                entries.clone(), copyIndices()
        );
    }


    @Override
    public T roundToZero() {
        return makeTensor(
                shape.copy(),
                ComplexOperations.roundToZero(this.entries, DEFAULT_ROUND_TO_ZERO_THRESHOLD),
                copyIndices()
        );
    }


    @Override
    public T roundToZero(double threshold) {
        return makeTensor(
                shape.copy(),
                ComplexOperations.roundToZero(this.entries, threshold),
                copyIndices()
        );
    }
}
