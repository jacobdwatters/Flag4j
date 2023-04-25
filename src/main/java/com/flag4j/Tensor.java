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
import com.flag4j.core.RealTensorBase;
import com.flag4j.core.TensorExclusiveMixin;
import com.flag4j.io.PrintOptions;
import com.flag4j.operations.TransposeDispatcher;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.operations.dense.complex.ComplexDenseOperations;
import com.flag4j.operations.dense.real.*;
import com.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import com.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import com.flag4j.operations.dense.real_complex.RealComplexDenseEquals;
import com.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;
import com.flag4j.util.StringUtils;

import java.util.Arrays;


/**
 * Real Dense Tensor. May have any rank (that is, may have any number of unique axes/dimensions).
 */
public class Tensor extends RealTensorBase<Tensor, CTensor>
        implements TensorExclusiveMixin<Tensor, Tensor, SparseTensor> {


    /**
     * Constructs a tensor with given shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public Tensor(Shape shape) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public Tensor(Shape shape, double fillValue) {
        super(shape, new double[shape.totalEntries().intValue()]);

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = fillValue;
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public Tensor(Shape shape, double[] entries) {
        super(shape, entries);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public Tensor(Shape shape, int[] entries) {
        super(shape, Arrays.stream(entries).asDoubleStream().toArray());

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public Tensor(Shape shape, Double[] entries) {
        super(shape, new double[entries.length]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        // Copy entries to tensor.
        int index = 0;
        for(Double value : entries) {
            super.entries[index++] = value;
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public Tensor(Shape shape, Integer[] entries) {
        super(shape, new double[entries.length]);

        if(entries.length != super.totalEntries().intValueExact()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        // Copy entries to tensor.
        int index = 0;
        for(Integer value : entries) {
            super.entries[index++] = value;
        }
    }


    /**
     * Constructs a tensor from another tensor. This effectively copies the tensor.
     * @param A tensor to copy.
     */
    public Tensor(Tensor A) {
        super(A.shape.copy(), A.entries.clone());
    }


    /**
     * Constructs a tensor whose shape and entries are specified by a matrix.
     * @param A Matrix to copy to tensor.
     */
    public Tensor(Matrix A) {
        super(A.shape.copy(), A.entries.clone());
    }


    /**
     * Constructs a tensor whose shape and entries are specified by a vector.
     * @param A Vector to copy to tensor.
     */
    public Tensor(Vector A) {
        super(A.shape.copy(), A.entries.clone());
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return ArrayUtils.isZeros(entries);
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return RealDenseProperties.isOnes(entries);
    }


    /**
     * Checks if an object is equal to this tensor object. Valid object types are: {@link Tensor}, {@link CTensor},
     * {@link SparseTensor}, and {@link SparseCTensor}. These tensors are equal to this tensor if all entries are
     * numerically equal to the corresponding element of this tensor. If the tensor is complex, then the imaginary
     * component must be zero to be equal.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors are numerically equivalent and false otherwise.
     */
    @Override
    public boolean equals(Object object) {
        boolean equal;

        if(object instanceof Tensor) {
            Tensor tensor = (Tensor) object;
            equal = RealDenseEquals.tensorEquals(this, tensor);
        } else if(object instanceof CTensor) {
            CTensor tensor = (CTensor) object;
            equal = RealComplexDenseEquals.tensorEquals(this, tensor);

        } else if(object instanceof SparseTensor) {
            SparseTensor tensor = (SparseTensor) object;
            equal = RealDenseSparseEquals.tensorEquals(this, tensor);

        } else if(object instanceof SparseCTensor) {
            SparseCTensor tensor = (SparseCTensor) object;
            equal = RealComplexDenseSparseEquals.tensorEquals(this, tensor);

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Creates a hashcode for this matrix. Note, method adds {@link Arrays#hashCode(double[])} applied on the
     * underlying data array and the underlying shape array.
     * @return The hashcode for this matrix.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(entries)+Arrays.hashCode(shape.dims);
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CTensor toComplex() {
        return new CTensor(this);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException If the number of indices provided does not match the rank of this tensor.
     * @throws IllegalArgumentException If any of the indices are outside the tensor for that respective axis.
     */
    @Override
    public Tensor set(double value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(indices.length, shape.getRank());
        RealDenseSetOperations.set(entries, shape, value, indices);
        return this;
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
    public Tensor reshape(Shape shape) {
        ParameterChecks.assertBroadcastable(this.shape, shape);
        return new Tensor(shape.copy(), this.entries.clone());
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
    public Tensor reshape(int... shape) {
        return reshape(new Shape(shape));
    }


    /**
     * Flattens tensor to single dimension.
     *
     * @return The flattened tensor.
     */
    @Override
    public Tensor flatten() {
        return new Tensor(new Shape(entries.length), this.entries.clone());
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Tensor add(Tensor B) {
        return new Tensor(
                shape.copy(),
                RealDenseOperations.add(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Tensor add(double a) {
        return new Tensor(
                shape.copy(),
                RealDenseOperations.add(entries, a)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CTensor add(CNumber a) {
        return new CTensor(
                shape.copy(),
                ComplexDenseOperations.add(entries, a)
        );
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    @Override
    public void addEq(Tensor B) {
        RealDenseOperations.addEq(entries, shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    @Override
    public void addEq(SparseTensor B) {
        RealDenseSparseOperations.addEq(this, B);
    }


    /**
     * Adds a specified value to all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to add to all entries of this tensor.
     */
    @Override
    public void addEq(Double b) {
        RealDenseOperations.addEq(entries, b);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Tensor sub(Tensor B) {
        return new Tensor(
                shape.copy(),
                RealDenseOperations.sub(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Tensor sub(double a) {
        return new Tensor(
                shape.copy(),
                RealDenseOperations.sub(entries, a)
        );
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CTensor sub(CNumber a) {
        return new CTensor(
                shape.copy(),
                ComplexDenseOperations.sub(entries, a)
        );
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(Tensor B) {
        RealDenseOperations.subEq(entries, shape, B.entries, B.shape);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(Double b) {
        RealDenseOperations.subEq(entries, b);
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public Tensor mult(double factor) {
        return new Tensor(shape.copy(),
                RealOperations.scalMult(entries, factor)
        );
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CTensor mult(CNumber factor) {
        return new CTensor(shape.copy(),
                ComplexOperations.scalMult(entries, factor)
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
    public Tensor div(double divisor) {
        return new Tensor(
                shape.copy(),
                RealDenseOperations.scalDiv(entries, divisor)
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
    public CTensor div(CNumber divisor) {
        return new CTensor(shape.copy(),
                RealComplexDenseOperations.scalDiv(entries, divisor)
        );
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        return AggregateReal.sum(entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public Tensor sqrt() {
        return new Tensor(
                shape.copy(),
                RealOperations.sqrt(entries)
        );
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public Tensor abs() {
        return new Tensor(
                shape.copy(),
                RealOperations.abs(entries)
        );
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     * In the context of a tensor, this exchanges the first and last axis of the tensor.
     * Also see {@link #transpose(int, int)} and {@link #T(int, int)}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Tensor transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * In the context of a tensor, this exchanges the first and last axis of the tensor.
     * Also see {@link #transpose(int, int)} and {@link #T(int, int)}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Tensor T() {
        return TransposeDispatcher.dispatchTensor(this, 0, shape.getRank()-1);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified axes. That is,
     * computes the sum of products between the two tensors along the specified axes.
     *
     * @param src2  Tensor to contract with this tensor.
     * @param aAxis Axis along which to compute products for this tensor.
     * @param bAxis Axis along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes {@code aAxis}
     *                                  and {@code bAxis}.
     * @throws IllegalArgumentException If either axis is out of bounds of the corresponding tensor.
     */
    @Override
    public Tensor tensorDot(Tensor src2, int aAxis, int bAxis) {
        return RealDenseTensorDot.tensorDot(this, src2, new int[]{aAxis}, new int[]{bAxis});
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2  Tensor to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public Tensor tensorDot(Tensor src2, int[] aAxes, int[] bAxes) {
        return RealDenseTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     *
     * @param src2 Tensor to compute dot product with this tensor.
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     *                                  along the second-to-last axis.
     */
    @Override
    public Tensor dot(Tensor src2) {
        return RealDenseTensorDot.dot(this, src2);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link #transpose()} and {@link #T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor.
     */
    @Override
    public Tensor transpose(int axis1, int axis2) {
        return T(axis1, axis2);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link #transpose()} and {@link #T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor.
     */
    @Override
    public Tensor T(int axis1, int axis2) {
        return TransposeDispatcher.dispatchTensor(this, axis1, axis2);
    }


    /**
     * Computes the transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation. Same as {@link #T(int[])}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public Tensor transpose(int... axes) {
        return T(axes);
    }


    /**
     * Computes the transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation. Same as {@link #transpose(int[])}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public Tensor T(int... axes) {
        // TODO: Add dispatcher for this method to choose between concurrent and sequential implementations.
        return new Tensor(
                shape.copy().swapAxes(axes),
                RealDenseTranspose.standardConcurrent(this.entries, this.shape, axes)
        );
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Tensor add(SparseTensor B) {
        return RealDenseSparseOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensor add(CTensor B) {
        return new CTensor(
                shape.copy(),
                RealComplexDenseOperations.add(B.entries, B.shape, this.entries, this.shape)
        );
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensor add(SparseCTensor B) {
        return RealComplexDenseSparseOperations.add(this, B);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Tensor sub(SparseTensor B) {
        return RealDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensor sub(CTensor B) {
        return new CTensor(
                shape.copy(),
                RealComplexDenseOperations.sub(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensor sub(SparseCTensor B) {
        return RealComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(SparseTensor B) {
        RealDenseSparseOperations.subEq(this, B);
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public Tensor recip() {
        return new Tensor(
                shape.copy(),
                RealDenseOperations.recip(entries)
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
    public Double get(int... indices) {
        ParameterChecks.assertArrayLengthsEq(indices.length, shape.getRank());
        return entries[shape.entriesIndex(indices)];
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public Tensor copy() {
        return new Tensor(this);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        if(this.entries.length==0) {
            return new int[]{};
        } else {
            return shape.getIndices(AggregateDenseReal.argMin(entries));
        }
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        if(this.entries.length==0) {
            return new int[]{};
        } else {
            return shape.getIndices(AggregateDenseReal.argMax(entries));
        }
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return RealDenseOperations.tensorNormL2(entries);
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
        return RealDenseOperations.tensorNormLp(entries, p);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public Tensor elemMult(Tensor B) {
        return new Tensor(
                shape.copy(),
                RealDenseElemMult.dispatch(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public SparseTensor elemMult(SparseTensor B) {
        return RealDenseSparseOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemMult(CTensor B) {
        return new CTensor(
                this.shape.copy(),
                RealComplexDenseElemMult.dispatch(B.entries, B.shape, this.entries, this.shape)
        );
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public SparseCTensor elemMult(SparseCTensor B) {
        return RealComplexDenseSparseOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public Tensor elemDiv(Tensor B) {
        return new Tensor(
                shape.copy(),
                RealDenseElemDiv.dispatch(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise division between two tensors.
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemDiv(CTensor B) {
        return new CTensor(
                shape.copy(),
                RealComplexDenseElemDiv.dispatch(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public Vector toVector() {
        return new Vector(this.entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank is larger than 2, it will
     * be flattened to a single row.
     */
    public Matrix toMatrix() {
        Matrix mat;

        if(this.getRank()==2) {
            mat = new Matrix(this.shape.copy(), this.entries.clone());
        } else {
            mat = new Matrix(1, this.entries.length, this.entries.clone());
        }

        return mat;
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flatten entries of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = shape.totalEntries().intValueExact();
        StringBuilder result = new StringBuilder(String.format("Full Shape: %s\n", shape));
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
