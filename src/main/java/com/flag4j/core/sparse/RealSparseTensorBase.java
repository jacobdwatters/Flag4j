/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.RealTensorMixin;
import com.flag4j.core.TensorBase;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.operations.dense.real.AggregateDenseReal;
import com.flag4j.operations.dense.real.RealDenseOperations;
import com.flag4j.operations.dense.real.RealDenseProperties;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;
import com.flag4j.operations.sparse.coo.SparseDataWrapper;

import java.math.BigInteger;

/**
 * Base class for all sparse tensor.
 * @param <T> Type of this tensor.
 * @param <U> Dense Tensor type.
 * @param <W> Complex Tensor type.
 * @param <Z> Dense complex tensor type.
 */
public abstract class RealSparseTensorBase<
        T extends TensorBase<T, ?, ?, ?, ?, ?, ?>,
        U, W, Z>
        extends SparseTensorBase<T, U, W, Z, T, double[], Double>
        implements RealTensorMixin<T, W> {

    /**
     * Creates a sparse tensor with specified shape. Note, this constructor stores indices for each element in the
     * <b>same</b> array. That is, for a shape with rank {@code m} and {@code n} non-zero entries,
     * the indices array will have shape {@code n-by-m}.
     * This is the opposite of {@link #RealSparseTensorBase(Shape, int, double[], int[], int[][])}.
     * @param shape Shape of this tensor.
     * @param nonZeroEntries Number of non-zero entries in the sparse tensor.
     * @param entries Non-zero entries of this sparse tensor.
     * @param indices Indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of rows in the indices array is not equal to the number of
     * elements in the entries array.
     * @throws IllegalArgumentException If the number of columns in the entries array is not equal to the rank of this
     * tensor.
     */
    protected RealSparseTensorBase(Shape shape, int nonZeroEntries, double[] entries, int[][] indices) {
        super(shape, nonZeroEntries, entries, indices);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }

        ParameterChecks.assertArrayLengthsEq(nonZeroEntries, indices.length);
    }


    /**
     * Creates a sparse tensor with specified shape. Note, this constructor stores indices for each element in different
     * arrays. That is, for a shape with rank {@code m} and {@code n} non-zero entries, the indices array will have shape
     * {@code m-by-n}. This is the opposite of {@link #RealSparseTensorBase(Shape, int, double[], int[][])}.
     * @param shape Shape
     * @param nonZeroEntries The number of non-zero entries of the tensor.
     * @param entries Non-zero entries of the sparse tensor.
     * @param initIndices Non-zero indices of the first axis of the tensor.
     * @param restIndices Non-zero indices of the rest of this tensor's axes.
     */
    protected RealSparseTensorBase(Shape shape, int nonZeroEntries, double[] entries, int[] initIndices, int[]... restIndices) {
        super(shape, nonZeroEntries, entries, initIndices, restIndices);

        if(super.totalEntries().compareTo(BigInteger.valueOf(nonZeroEntries)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, nonZeroEntries));
        }
    }


    /**
     * Creates a deep copy of the indices of this sparse tensor.
     * @return A deep copy of the indices of this sparse tensor.
     */
    private int[][] copyIndices() {
        int[][] newIndices = new int[indices.length][];
        ArrayUtils.deepCopy(indices, newIndices);
        return newIndices;
    }


    /**
     * A factory for creating a real sparse tensor.
     * @param shape Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    protected abstract T makeTensor(Shape shape, double[] entries, int[][] indices);


    /**
     * A factory for creating a real dense tensor.
     * @param shape Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    protected abstract U makeDenseTensor(Shape shape, double[] entries);


    /**
     * A factory for creating a complex sparse tensor.
     * @param shape Shape of the tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    protected abstract W makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices);


    /**
    * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
    */
    @Override
    public void sortIndices() {
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
    return AggregateReal.min(entries);
    }


    @Override
    public double max() {
    return AggregateReal.max(entries);
    }


    @Override
    public double minAbs() {
    return AggregateReal.minAbs(entries);
    }


    @Override
    public double maxAbs() {
    return AggregateReal.maxAbs(entries);
    }


    @Override
    public double infNorm() {
    return AggregateReal.maxAbs(entries);
    }


    @Override
    public boolean isPos() {
    return RealProperties.isPos(entries);
    }


    @Override
    public boolean isNeg() {
    return RealProperties.isNeg(entries);
    }


    @Override
    public boolean isZeros() {
     return entries.length==0 || RealProperties.isZeros(entries);
    }


    /**
     * Checks if this sparse tensors non-zero values are all ones.
     * @return True if this sparse tensors non-zero values are all ones. False otherwise.
     */
    @Override
    public boolean isOnes() {
        return RealDenseProperties.isOnes(entries);
    }


    @Override
    public Double sum() {
        return AggregateReal.sum(entries);
    }


    @Override
    public T mult(double factor) {
        return makeTensor(shape.copy(), RealOperations.scalMult(entries, factor), copyIndices());
    }


    @Override
    public W mult(CNumber factor) {
        return makeComplexTensor(shape.copy(), ComplexOperations.scalMult(entries, factor), copyIndices());
    }


    @Override
    public T div(double divisor) {
        return makeTensor(shape.copy(), RealOperations.scalMult(entries, divisor), copyIndices());
    }


    @Override
    public W div(CNumber divisor) {
        return makeComplexTensor(shape.copy(), ComplexOperations.scalMult(entries, divisor), copyIndices());
    }


    @Override
    public T sqrt() {
        return makeTensor(shape.copy(), RealOperations.sqrt(entries), copyIndices());
    }


    @Override
    public T abs() {
        return makeTensor(shape.copy(), RealOperations.abs(entries), copyIndices());
    }


    @Override
    public T reshape(int... dims) {
        return reshape(new Shape(dims));
    }


    @Override
    public T reshape(Shape newShape) {
        ParameterChecks.assertBroadcastable(shape, newShape);
        return makeTensor(newShape, entries.clone(), copyIndices());
    }


    @Override
    public T flatten() {
        return makeTensor(
                new Shape(shape.totalEntries().intValueExact()),
                entries.clone(), copyIndices()
        );
    }


    @Override
    public int[] argMax() {
        int idx = AggregateDenseReal.argMax(entries);
        return indices[idx].clone();
    }


    @Override
    public int[] argMin() {
        int idx = AggregateDenseReal.argMin(entries);
        return indices[idx].clone();
    }


    @Override
    public T recip() {
        return makeTensor(shape.copy(), RealDenseOperations.recip(entries), copyIndices());
    }


    @Override
    public T round() {
        return makeTensor(shape.copy(), RealOperations.round(this.entries), copyIndices());
    }


    @Override
    public T round(int precision) {
        return makeTensor(shape.copy(), RealOperations.round(this.entries, precision), copyIndices());
    }


    @Override
    public T roundToZero() {
        return makeTensor(
                shape.copy(),
                RealOperations.roundToZero(this.entries, DEFAULT_ROUND_TO_ZERO_THRESHOLD),
                copyIndices()
        );
    }


    @Override
    public T roundToZero(double threshold) {
        return makeTensor(
                shape.copy(),
                RealOperations.roundToZero(this.entries, threshold),
                copyIndices()
        );
    }
}
