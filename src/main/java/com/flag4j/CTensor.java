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
import com.flag4j.core.ComplexDenseTensorBase;
import com.flag4j.core.ComplexTensorExclusiveMixin;
import com.flag4j.operations.TransposeDispatcher;
import com.flag4j.operations.common.complex.AggregateComplex;
import com.flag4j.operations.dense.complex.*;
import com.flag4j.operations.dense.real_complex.RealComplexDenseEquals;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseOperations;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * Complex dense tensor.
 */
public class CTensor
        extends ComplexDenseTensorBase<CTensor, Tensor>
        implements ComplexTensorExclusiveMixin<CTensor> {


    /**
     * Constructs a tensor with given shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public CTensor(Shape shape) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        ArrayUtils.fillZeros(super.entries);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, CNumber fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, int[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * Note, unlike other constructors, the entries parameter is not copied.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, CNumber[] entries) {
        super(shape, entries);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(Tensor A) {
        super(A.shape.copy(), new CNumber[A.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(CTensor A) {
        super(A.shape.copy(), new CNumber[A.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CTensor makeTensor(Shape shape, CNumber[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CTensor getSelf() {
        return this;
    }


    /**
     * Factory to create a real tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected Tensor makeRealTensor(Shape shape, double[] entries) {
        return new Tensor(shape, entries);
    }


    /**
     * Flattens a tensor along the specified axis. The resulting tensor will have the same rank but only have values
     * along the single specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CTensor flatten(int axis) {
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(dims);

        return new CTensor(flatShape, entries.clone());
    }


    /**
     * Checks if an object is equal to this tensor object. Valid object types are: {@link Tensor}, {@link CTensor},
     * {@link SparseTensor}, and {@link SparseCTensor}. These tensors are equal to this tensor if all entries are
     * numerically equal to the corresponding element of this tensor.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors are numerically equivalent and false otherwise.
     */
    @Override
    public boolean equals(Object object) {
        boolean equal;

        if(object instanceof Tensor) {
            Tensor tensor = (Tensor) object;
            equal = RealComplexDenseEquals.tensorEquals(tensor, this);
        } else if(object instanceof CTensor) {
            CTensor tensor = (CTensor) object;
            equal = ComplexDenseEquals.tensorEquals(entries, shape, tensor.entries, tensor.shape);

        } else if(object instanceof SparseTensor) {
            SparseTensor tensor = (SparseTensor) object;
            equal = RealComplexDenseSparseEquals.tensorEquals(this, tensor);

        } else if(object instanceof SparseCTensor) {
            SparseCTensor tensor = (SparseCTensor) object;
            equal = ComplexDenseSparseEquals.tensorEquals(this, tensor);

        } else {
            equal = false;
        }

        return equal;
    }



    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CTensor hermTranspose() {
        return H();
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CTensor H() {
        return new CTensor(
                shape.copy().swapAxes(0, getRank()-1),
                ComplexDenseTranspose.standardConcurrentHerm(
                    entries,
                    shape,
                    0,
                    getRank()-1
                )
        );
    }


    /**
     * Computes the conjugate transpose of a tensor. Same as {@link #H(int, int)}.
     * In the context of a tensor, this exchanges the specified axes and takes the complex conjugate of elements along
     * those axes.
     * Also see {@link #hermTranspose() hermTranspose()} and
     * {@link #H() H()} to conjugate transpose first and last axes.
     *
     * @param axis1 First axis to exchange and apply complex conjugate.
     * @param axis2 Second axis to exchange and apply complex conjugate.
     * @return The conjugate transpose of this tensor.
     */
    @Override
    public CTensor hermTranspose(int axis1, int axis2) {
        return H(axis1, axis2);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #hermTranspose(int, int)}.
     * In the context of a tensor, this exchanges the specified axes and takes the complex conjugate of elements along
     * those axes.
     * Also see {@link #hermTranspose()} and
     * {@link #H()} to conjugate transpose first and last axes.
     *
     * @param axis1 First axis to exchange and apply complex conjugate.
     * @param axis2 Second axis to exchange and apply complex conjugate.
     * @return The conjugate transpose of this tensor.
     */
    @Override
    public CTensor H(int axis1, int axis2) {
        return new CTensor(
                shape.copy().swapAxes(axis1, axis2),
                ComplexDenseTranspose.standardConcurrentHerm(
                        entries,
                        shape,
                        axis1,
                        axis2
                )
        );
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation and takes the complex conjugate of the elements of these axes. Same as {@link #H(int[])}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public CTensor hermTranspose(int... axes) {
        return H(axes);
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation and takes the complex conjugate of the elements of these axes. Same as {@link #hermTranspose(int[])}}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public CTensor H(int... axes) {
        return new CTensor(
                this.shape.copy().swapAxes(axes),
                ComplexDenseTranspose.standardConcurrentHerm(this.entries, this.shape, axes)
        );
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link #transpose() transpose()} and
     * {@link #T() T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor.
     */
    @Override
    public CTensor transpose(int axis1, int axis2) {
        return T(axis1, axis2);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link #transpose()} and
     * {@link #T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     * @return The transpose of this tensor.
     */
    @Override
    public CTensor T(int axis1, int axis2) {
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
    public CTensor transpose(int... axes) {
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
    public CTensor T(int... axes) {
        // TODO: Add dispatcher for this method to choose between concurrent and sequential implementations.
        return new CTensor(
                this.shape.copy().swapAxes(axes),
                ComplexDenseTranspose.standardConcurrent(this.entries, this.shape, axes)
        );
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
    public CTensor tensorDot(CTensor src2, int aAxis, int bAxis) {
        return ComplexDenseTensorDot.tensorDot(this, src2, new int[]{aAxis}, new int[]{bAxis});
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
    public CTensor tensorDot(CTensor src2, int[] aAxes, int[] bAxes) {
        return ComplexDenseTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
    public CTensor dot(CTensor src2) {
        return ComplexDenseTensorDot.dot(this, src2);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensor add(SparseTensor B) {
        return new CTensor(RealComplexDenseSparseOperations.add(this, B));
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
        return ComplexDenseSparseOperations.add(this, B);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensor sub(SparseTensor B) {
        return RealComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CTensor transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CTensor T() {
        return TransposeDispatcher.dispatchTensor(this, 0, shape.getRank()-1);
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
        return ComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(SparseTensor B) {
        RealComplexDenseSparseOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(SparseTensor B) {
        RealComplexDenseSparseOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public SparseCTensor elemMult(SparseTensor B) {
        return RealComplexDenseSparseOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public SparseCTensor elemMult(SparseCTensor B) {
        return ComplexDenseSparseOperations.elemMult(this, B);
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public CTensor flatten() {
        return new CTensor(new Shape(entries.length), this.entries.clone());
    }

    
    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return ComplexDenseOperations.tensorNormL2(entries);
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
        return ComplexDenseOperations.tensorNormLp(entries, p);
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double infNorm() {
        return AggregateComplex.maxAbs(entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public CVector toVector() {
        CNumber[] entries = new CNumber[this.entries.length];
        ArrayUtils.copy2CNumber(this.entries, entries);

        return new CVector(entries);
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank is larger than 2, it will
     * be flattened to a single row.
     */
    public CMatrix toMatrix() {
        CMatrix mat;

        CNumber[] entries = new CNumber[this.entries.length];
        ArrayUtils.copy2CNumber(this.entries, entries);

        if(this.getRank()==2) {
            mat = new CMatrix(this.shape.copy(), entries);
        } else {
            mat = new CMatrix(1, this.entries.length, entries);
        }

        return mat;
    }
}
