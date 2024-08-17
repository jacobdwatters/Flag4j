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

import org.flag4j.arrays_old.sparse.CooCTensor;
import org.flag4j.arrays_old.sparse.CooTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.core.TensorExclusiveMixin;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.TensorInvert;
import org.flag4j.operations_old.TransposeDispatcher;
import org.flag4j.operations_old.dense.real.RealDenseEquals;
import org.flag4j.operations_old.dense.real.RealDenseTensorDot;
import org.flag4j.operations_old.dense.real.RealDenseTranspose;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations_old.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations_old.dense_sparse.coo.real.RealDenseSparseTensorOperations;
import org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

import java.util.Arrays;

// TODO: Allow for zero dimension shapes for scalar tensors.
/**
 * Real Dense TensorOld. May have any rank (that is, may have any number of unique axes/dimensions).
 */
@Deprecated
public class TensorOld
        extends RealDenseTensorBase<TensorOld, CTensorOld>
        implements TensorExclusiveMixin<TensorOld, TensorOld, CooTensor, CTensorOld> {


    /**
     * Constructs a tensor with given shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public TensorOld(Shape shape) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
        shape.makeStridesIfNull();
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public TensorOld(Shape shape, double fillValue) {
        super(shape, new double[shape.totalEntries().intValueExact()]);

        for(int i=0; i<super.totalEntries().intValueExact(); i++) {
            super.entries[i] = fillValue;
        }

        shape.makeStridesIfNull();
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public TensorOld(Shape shape, double[] entries) {
        super(shape, entries);
        shape.makeStridesIfNull();

        if(entries.length != super.totalEntries().intValueExact()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public TensorOld(Shape shape, int[] entries) {
        super(shape, Arrays.stream(entries).asDoubleStream().toArray());
        shape.makeStridesIfNull();

        if(entries.length != super.totalEntries().intValueExact()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public TensorOld(Shape shape, Double[] entries) {
        super(shape, new double[entries.length]);
        shape.makeStridesIfNull();

        if(entries.length != super.totalEntries().intValueExact()) {
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
    public TensorOld(Shape shape, Integer[] entries) {
        super(shape, new double[entries.length]);
        shape.makeStridesIfNull();

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
    public TensorOld(TensorOld A) {
        super(A.shape, A.entries.clone());
        shape.makeStridesIfNull();
    }


    /**
     * Constructs a tensor whose shape and entries are specified by a matrix.
     * @param A MatrixOld to copy to tensor.
     */
    public TensorOld(MatrixOld A) {
        super(A.shape, A.entries.clone());
        shape.makeStridesIfNull();
    }


    /**
     * Constructs a tensor whose shape and entries are specified by a vector.
     * @param A VectorOld to copy to tensor.
     */
    public TensorOld(VectorOld A) {
        super(A.shape, A.entries.clone());
        shape.makeStridesIfNull();
    }


    /**
     * Factory to create a tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected TensorOld makeTensor(Shape shape, double[] entries) {
        shape.makeStridesIfNull();
        return new TensorOld(shape, entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CTensorOld makeComplexTensor(Shape shape, double[] entries) {
        shape.makeStridesIfNull();
        return new CTensorOld(shape, entries);
    }


    /**
     * Factory to create a complex tensor with the specified shape and size.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the tensor to make.
     * @return A new tensor with the specified shape and entries.
     */
    @Override
    protected CTensorOld makeComplexTensor(Shape shape, CNumber[] entries) {
        shape.makeStridesIfNull();
        return new CTensorOld(shape, entries);
    }


    /**
     * Simply returns this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected TensorOld getSelf() {
        return this;
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public TensorOld flatten(int axis) {
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(true, dims);

        return new TensorOld(flatShape, entries.clone());
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link TensorOld}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        TensorOld src2 = (TensorOld) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor which is equivalent to this dense tensor.
     */
    @Override
    public CooTensor toCoo() {
        return CooTensor.fromDense(this);
    }


    /**
     * Flattens tensor to single dimension.
     *
     * @return The flattened tensor.
     */
    @Override
    public TensorOld flatten() {
        return new TensorOld(new Shape(true, entries.length), this.entries.clone());
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    public void addEq(CooTensor B) {
        RealDenseSparseTensorOperations.addEq(this, B);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * In the context of a tensor, this exchanges the first and last axis of the tensor.
     * Also see {@link #transpose(int, int)} and {@link #T(int, int)}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public TensorOld T() {
        return TransposeDispatcher.dispatchTensor(this, 0, shape.getRank()-1);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2  TensorOld to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public TensorOld tensorDot(TensorOld src2, int[] aAxes, int[] bAxes) {
        return RealDenseTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     *
     * @param src2 TensorOld to compute dot product with this tensor.
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     *                                  along the second-to-last axis.
     * @see #tensorDot(TensorOld, int[], int[])
     */
    @Override
    public TensorOld tensorDot(TensorOld src2) {
        return RealDenseTensorDot.tensorDot(this, src2);
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
    public TensorOld T(int axis1, int axis2) {
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
    public TensorOld T(int... axes) {
        // TODO: Add dispatcher for this method to choose between concurrent and sequential implementations.
        return new TensorOld(
                shape.swapAxes(axes),
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
    public TensorOld add(CooTensor B) {
        return RealDenseSparseTensorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensorOld add(CTensorOld B) {
        return new CTensorOld(
                shape,
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
    public CTensorOld add(CooCTensor B) {
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
    public TensorOld sub(CooTensor B) {
        return RealDenseSparseTensorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensorOld sub(CTensorOld B) {
        return new CTensorOld(
                shape,
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
    public CTensorOld sub(CooCTensor B) {
        return RealComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    public void subEq(CooTensor B) {
        RealDenseSparseTensorOperations.subEq(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooTensor elemMult(CooTensor B) {
        return RealDenseSparseTensorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensorOld elemMult(CTensorOld B) {
        return new CTensorOld(
                this.shape,
                RealComplexDenseElemMult.dispatch(B.entries, B.shape, this.entries, this.shape)
        );
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensor elemMult(CooCTensor B) {
        return RealComplexDenseSparseOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     * @param B TensorOld to element-wise divide with this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensorOld elemDiv(CTensorOld B) {
        return new CTensorOld(
                shape,
                RealComplexDenseElemDiv.dispatch(entries, shape, B.entries, B.shape)
        );
    }


    /**
     * Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product
     * operation.
     * A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #tensorInv()
     */
    @Override
    public TensorOld tensorInv(int numIndices) {
        return TensorInvert.inv(this, numIndices);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public VectorOld toVector() {
        return new VectorOld(this.entries.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be broadcastable with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     */
    public MatrixOld toMatrix(Shape matShape) {
        ParameterChecks.assertBroadcastable(shape, matShape);
        ParameterChecks.assertRank(2, matShape);

        return new MatrixOld(matShape, entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank is larger than 2, it will
     * be flattened to a single row.
     */
    public MatrixOld toMatrix() {
        MatrixOld mat;

        if(this.getRank()==2) {
            mat = new MatrixOld(this.shape, this.entries.clone());
        } else {
            mat = new MatrixOld(1, this.entries.length, this.entries.clone());
        }

        return mat;
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flattened entries of this tensor.
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