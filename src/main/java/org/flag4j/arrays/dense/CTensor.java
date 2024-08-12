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

import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.ComplexTensorExclusiveMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.core.TensorExclusiveMixin;
import org.flag4j.core.dense_base.ComplexDenseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.TensorInvert;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.dense.complex.ComplexDenseEquals;
import org.flag4j.operations.dense.complex.ComplexDenseTensorDot;
import org.flag4j.operations.dense.complex.ComplexDenseTranspose;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

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
        shape.makeStridesIfNull();
        Arrays.fill(super.entries, CNumber.ZERO);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        shape.makeStridesIfNull();
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, CNumber fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        shape.makeStridesIfNull();
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        shape.makeStridesIfNull();

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
        shape.makeStridesIfNull();

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
        shape.makeStridesIfNull();
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(Tensor A) {
        super(A.shape, new CNumber[A.totalEntries().intValue()]);
        shape.makeStridesIfNull();
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(CTensor A) {
        super(A.shape, new CNumber[A.totalEntries().intValue()]);
        shape.makeStridesIfNull();
        System.arraycopy(A.entries, 0, super.entries, 0, A.entries.length);
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
        shape.makeStridesIfNull();
        return new CTensor(shape, entries);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor which is equivalent to this dense tensor.
     */
    @Override
    public CooCTensor toCoo() {
        return CooCTensor.fromDense(this);
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
        shape.makeStridesIfNull();
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
        Shape flatShape = new Shape(true, dims);

        return new CTensor(flatShape, entries.clone());
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CTensor src2 = (CTensor) object;

        return ComplexDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
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
                shape.swapAxes(0, getRank()-1),
                ComplexDenseTranspose.standardConcurrentHerm(
                    entries,
                    shape,
                    0,
                    getRank()-1
                )
        );
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
                shape.swapAxes(axis1, axis2),
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
                this.shape.swapAxes(axes),
                ComplexDenseTranspose.standardConcurrentHerm(this.entries, this.shape, axes)
        );
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    public void addEq(Tensor B) {
        RealComplexDenseOperations.addEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    public void addEq(CooCTensor B) {
        ComplexDenseSparseOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    public void subEq(Tensor B) {
        RealComplexDenseOperations.subEq(this.entries, this.shape, B.entries, B.shape);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    public void subEq(CooCTensor B) {
        ComplexDenseSparseOperations.subEq(this, B);
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
                this.shape.swapAxes(axes),
                ComplexDenseTranspose.standardConcurrent(this.entries, this.shape, axes)
        );
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
    public CTensor tensorDot(CTensor src2) {
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
    public CTensor add(CooTensor B) {
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
    public CTensor add(Tensor B) {
        return new CTensor(
                this.shape,
                RealComplexDenseOperations.add(this.entries, this.shape, B.entries, B.shape)
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
    public CTensor sub(Tensor B) {
        return new CTensor(
                this.shape,
                RealComplexDenseOperations.sub(this.entries, this.shape, B.entries, B.shape)
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
    public CTensor add(CooCTensor B) {
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
    public CTensor sub(CooTensor B) {
        return RealComplexDenseSparseOperations.sub(this, B);
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
    public CTensor sub(CooCTensor B) {
        return ComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    public void addEq(CooTensor B) {
        RealComplexDenseSparseOperations.addEq(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    public void subEq(CooTensor B) {
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
    public CTensor elemMult(Tensor B) {
        return new CTensor(
                this.shape,
                RealComplexDenseElemMult.dispatch(this.entries, this.shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensor elemMult(CooTensor B) {
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
    public CooCTensor elemMult(CooCTensor B) {
        return ComplexDenseSparseOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide from this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemDiv(Tensor B) {
        return new CTensor(
                shape,
                RealComplexDenseElemDiv.dispatch(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) this.tensorDot(X, numIndices)} is the
     * 'identity' tensor for the tensor
     * dot product
     * operation.
     * A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #tensorInv()
     */
    @Override
    public CTensor tensorInv(int numIndices) {
        return TensorInvert.inv(this, numIndices);
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public CTensor flatten() {
        return new CTensor(new Shape(true, entries.length), this.entries.clone());
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public CVector toVector() {
        CNumber[] entries = new CNumber[this.entries.length];
        System.arraycopy(this.entries, 0, entries, 0, entries.length);

        return new CVector(entries);
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be broadcastable with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     */
    public CMatrix toMatrix(Shape matShape) {
        ParameterChecks.assertBroadcastable(shape, matShape);
        ParameterChecks.assertRank(2, matShape);

        return new CMatrix(matShape, Arrays.copyOf(entries, entries.length));
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
        System.arraycopy(this.entries, 0, entries, 0, entries.length);

        if(this.getRank()==2) {
            mat = new CMatrix(this.shape, entries);
        } else {
            mat = new CMatrix(1, this.entries.length, entries);
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
