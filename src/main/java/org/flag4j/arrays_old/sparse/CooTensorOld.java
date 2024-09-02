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

package org.flag4j.arrays_old.sparse;


import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.core.TensorExclusiveMixin;
import org.flag4j.core.sparse_base.RealSparseTensorBase;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.TensorInvert;
import org.flag4j.operations_old.dense.real.RealDenseTranspose;
import org.flag4j.operations_old.dense_sparse.coo.real.RealDenseSparseTensorOperations;
import org.flag4j.operations_old.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;
import org.flag4j.operations_old.sparse.coo.real.RealCooTensorDot;
import org.flag4j.operations_old.sparse.coo.real.RealCooTensorOperations;
import org.flag4j.operations_old.sparse.coo.real.RealSparseEquals;
import org.flag4j.operations_old.sparse.coo.real_complex.RealComplexCooTensorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Real sparse tensor. Can be any rank. Stored in coordinate (COO) format.
 */
@Deprecated
public class CooTensorOld
        extends RealSparseTensorBase<CooTensorOld, TensorOld, CooCTensorOld, CTensorOld>
        implements TensorExclusiveMixin<CooTensorOld, TensorOld, CooTensorOld, CooCTensorOld>
{


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public CooTensorOld(Shape shape) {
        super(shape, 0, new double[0], new int[0][0]);
        
    }


    /**
     * Creates a sparse tensor with specified shape and non-zero values/indices.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooTensorOld(Shape shape, double[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, indices);
        
    }


    /**
     * Creates a sparse tensor with specified shape and non-zero values/indices.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooTensorOld(Shape shape, int[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, ArrayUtils.asDouble(nonZeroEntries, null), indices);
        
    }


    /**
     * Creates a sparse tensor with specified shape and non-zero values/indices.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooTensorOld(Shape shape, List<Double> nonZeroEntries, List<int[]> indices) {
        super(shape, nonZeroEntries.size(),
                ArrayUtils.fromDoubleList(nonZeroEntries),
                indices.toArray(new int[0][]));
        
    }



    /**
     * Constructs a sparse tensor whose shape and non-zero values/indices are given by another sparse tensor.
     * This effectively copies the tensor.
     * @param A TensorOld to copy.
     */
    public CooTensorOld(CooTensorOld A) {
        super(A.shape, A.nonZeroEntries(), A.entries.clone(), new int[A.indices.length][A.indices[0].length]);
        for(int i=0; i<indices.length; i++) {
            super.indices[i] = A.indices[i].clone();
        }
    }


    /**
     * Checks if an object is equal to this sparse COO tensor.
     * @param object Object to compare this sparse COO tensor to.
     * @return True if the object is a {@link CooTensorOld}, has the same shape as this tensor, and is element-wise equal to this
     * tensor.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooTensorOld src2 = (CooTensorOld) object;
        return RealSparseEquals.tensorEquals(this, src2);
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CooCTensorOld toComplex() {
        return new CooCTensorOld(
                shape,
                ArrayUtils.copy2CNumber(entries, null),
                ArrayUtils.deepCopy(indices, null)
        );
    }


    /**
     * Converts a sparse {@link CooTensorOld} from a dense {@link TensorOld}. This is likely only worthwhile for very sparse tensors.
     * @param src Dense tensor to convert to sparse COO tensor.
     * @return A COO tensor which is equivalent to the {@code src} dense tensor.
     */
    public static CooTensorOld fromDense(TensorOld src) {
        List<Double> entries = new ArrayList<>();
        List<int[]> indices = new ArrayList<>();

        int size = src.entries.length;
        double value;

        for(int i=0; i<size; i++) {
            value = src.entries[i];

            if(value != 0) {
                entries.add(value);
                indices.add(src.shape.getIndices(i));
            }
        }

        return new CooTensorOld(src.shape, ArrayUtils.fromDoubleList(entries), indices.toArray(new int[0][]));
    }


    /**
     * A factory for creating a real sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CooTensorOld makeTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooTensorOld(shape, entries, indices);
    }


    /**
     * A factory for creating a real dense tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected TensorOld makeDenseTensor(Shape shape, double[] entries) {
        return new TensorOld(shape, entries);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CooCTensorOld makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CooCTensorOld(shape, entries, indices);
    }


    /**
     * <p>Copies this tensor and sets the specified {@code index} to {@code value}.</p>
     *
     * <p>Note: unlike with dense tensors, this method returns a copy of this
     * tensor with the specified value updated and does <b>NOT</b> return a reference to this tensor. Because of this, it may be
     * expensive to make frequent updates to a sparse tensor.</p>
     *
     * @param value Value to set.
     * @param index The index of this tensor for which to set the value.
     * @return A copy of this tensor with the specified value set.
     */
    @Override
    public CooTensorOld set(double value, int... index) {
        ParameterChecks.ensureValidIndex(shape, index);
        CooTensorOld dest;

        // Check if value already exists in tensor.
        int idx = -1;
        for(int i=0; i<indices.length; i++) {
            if(Arrays.equals(indices[i], index)) {
                idx = i;
                break; // Found in tensor, no need to continue.
            }
        }

        if(idx > -1) {
            // Copy entries and set new value.
            dest = new CooTensorOld(shape, entries.clone(), ArrayUtils.deepCopy(indices, null));
            dest.entries[idx] = value;
            dest.indices[idx] = index;
        } else {
            // Copy old indices and insert new one.
            int[][] newIndices = new int[indices.length + 1][getRank()];
            ArrayUtils.deepCopy(indices, newIndices);
            newIndices[indices.length] = index;

            // Copy old entries and insert new one.
            double[] newEntries = Arrays.copyOf(entries, entries.length+1);
            newEntries[newEntries.length-1] = value;

            // Ensure indices are sorted.
            dest = new CooTensorOld(shape, newEntries, newIndices);
            dest.sortIndices();
        }

        return dest;
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
    public CooTensorOld reshape(Shape newShape) {
        ParameterChecks.ensureBroadcastable(shape, newShape);

        int rank = indices[0].length;
        int newRank = newShape.getRank();
        int nnz = entries.length;

        int[] oldStrides = shape.getStrides();
        int[] newStrides = newShape.getStrides();

        int[][] newIndices = new int[nnz][newRank];

        for (int i = 0; i < nnz; i++) {
            int flatIndex = 0;
            for (int j = 0; j < rank; j++) {
                flatIndex += indices[i][j] * oldStrides[j];
            }

            for (int j = 0; j < newRank; j++) {
                int[] arr1 = newIndices[i];
                int v1 = newIndices[i][j];
                int v2 = newStrides[j];

                newIndices[i][j] = flatIndex / newStrides[j];
                flatIndex %= newStrides[j];
            }
        }

        return new CooTensorOld(newShape, entries.clone(), newIndices);
    }


    /**
     * Flattens tensor to single dimension. This method does not preserve rank. To flatten tensor along a single axis (and preserve
     * the rank) use {@link #flatten(int)}.
     *
     * @return The flattened tensor.
     * @see #flatten(int)
     */
    @Override
    public CooTensorOld flatten() {
        int[][] destIndices = new int[entries.length][1];

        for(int i=0, size=entries.length; i<size; i++)
            destIndices[i][0] = shape.entriesIndex(indices[i]);

        return new CooTensorOld(new Shape(shape.totalEntries().intValueExact()), entries.clone(), destIndices);
    }


    /**
     * Flattens a tensor along the specified axis. Unlike {@link #flatten()}, this method will preserve the rank of the tensor.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is negative or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public CooTensorOld flatten(int axis) {
        ParameterChecks.ensureIndexInBounds(indices[0].length, axis);
        int[][] destIndices = new int[indices.length][indices[0].length];

        // Compute new shape.
        int[] destShape = new int[indices[0].length];
        Arrays.fill(destShape, 1);
        destShape[axis] = shape.totalEntries().intValueExact();

        for(int i=0, size=entries.length; i<size; i++)
            destIndices[i][axis] = shape.entriesIndex(indices[i]);

        return new CooTensorOld(new Shape(destShape), entries.clone(), destIndices);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public TensorOld tensorDot(CooTensorOld src2, int[] aAxes, int[] bAxes) {
        return RealCooTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose(int, int)}.
     * In the context of a tensor, this exchanges the specified axes.
     * Also see {@link #transpose()} and
     * {@link #T()} to exchange first and last axes.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooTensorOld T(int axis1, int axis2) {
        int rank = getRank();
        ParameterChecks.ensureIndexInBounds(rank, axis1, axis2);

        if(axis1 == axis2) return copy(); // Simply return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        double[] transposeEntries = new double[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i];
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create sparse coo tensor and sort values lexicographically.
        CooTensorOld transpose = new CooTensorOld(shape.swapAxes(axis1, axis2), transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Computes the transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation. Same as {@link #transpose(int[])}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public CooTensorOld T(int... axes) {
        int rank = getRank();
        ParameterChecks.ensureEquals(rank, axes.length);
        ParameterChecks.ensurePermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        double[] transposeEntries = new double[nnz];

        // Permute the indices according to the permutation array.
        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i];
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create sparse coo tensor and sort values lexicographically.
        CooTensorOld transpose = new CooTensorOld(shape.swapAxes(axes), transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooTensorOld add(CooTensorOld B) {
        return RealCooTensorOperations.add(this, B);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public TensorOld add(TensorOld B) {
        return RealDenseSparseTensorOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public TensorOld sub(TensorOld B) {
        return RealDenseSparseTensorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensorOld add(CTensorOld B) {
        return RealComplexDenseSparseOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooCTensorOld add(CooCTensorOld B) {
        return RealComplexCooTensorOperations.add(B, this);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public TensorOld add(double a) {
        return RealDenseSparseTensorOperations.add(this, a);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CTensorOld add(CNumber a) {
        return RealComplexDenseSparseOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooTensorOld sub(CooTensorOld B) {
        return RealCooTensorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CTensorOld sub(CTensorOld B) {
        return RealComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooCTensorOld sub(CooCTensorOld B) {
        return RealComplexCooTensorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B TensorOld to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooTensorOld elemMult(TensorOld B) {
        return RealDenseSparseTensorOperations.elemMult(B, this);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public TensorOld sub(double a) {
        return RealDenseSparseTensorOperations.sub(this, a);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CTensorOld sub(CNumber a) {
        return RealComplexDenseSparseOperations.sub(this, a);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     * In the context of a tensor, this exchanges the first and last axis of the tensor.
     * Also see {@link #transpose(int, int)} and {@link #T(int, int)}.
     *
     * @return The transpose of this tensor.
     * @see #T()
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public CooTensorOld transpose() {
        return T(0, getRank()-1);
    }


    /**
     * <p>Computes the transpose of a tensor. Same as {@link #transpose()}.</p>
     *
     * <p>This method transposes the tensor by exchanges the first and last index
     * of the tensor. Thus, for a rank 2 tensor, this method is equivalent to a matrix transpose.</p>
     *
     * <p>{@link #T(int, int)} and {@link #T(int...)} offer more general tensor transposes.</p>
     *
     * @return The transpose of this tensor.
     * @see #transpose()
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public CooTensorOld T() {
        return T(0, getRank()-1);
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
        ParameterChecks.ensureValidIndex(shape, indices);

        for(int i = 0; i < nnz; i++) {
            if(Arrays.equals(this.indices[i], indices)) {
                return entries[i];
            }
        }

        return 0.0; // Return zero if the index is not found
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CooTensorOld copy() {
        return new CooTensorOld(shape, entries.clone(), ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B TensorOld to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooTensorOld elemMult(CooTensorOld B) {
        return RealCooTensorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B TensorOld to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensorOld elemMult(CTensorOld B) {
        return RealComplexDenseSparseOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B TensorOld to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensorOld elemMult(CooCTensorOld B) {
        return RealComplexCooTensorOperations.elemMult(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B TensorOld to element-wise divide with this tensor.
     *
     * @return The result of the element-wise tensor division.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensorOld elemDiv(CTensorOld B) {
        return RealComplexDenseSparseOperations.elemDiv(this, B);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B TensorOld to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooTensorOld elemDiv(TensorOld B) {
        return RealDenseSparseTensorOperations.elemDiv(this, B);
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link #tensorDot(TensorBase, int) this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product operation.
     * A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.</p>
     *
     * <p>WARNING: This method will convert this tensor to a dense tensor.</p>
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     *
     * @return The 'inverse' of this tensor as defined in the above sense.
     *
     * @see #tensorInv()
     */
    @Override
    public TensorOld tensorInv(int numIndices) {
        return TensorInvert.inv(toDense(), numIndices);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CooTensorOld getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooTensorOld tensor, double relTol, double absTol) {
        return RealSparseEquals.allCloseTensor(this, tensor, relTol, absTol);
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be broadcastable with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     */
    public CooMatrixOld toMatrix(Shape matShape) {
        ParameterChecks.ensureRank(matShape, 2);
        CooTensorOld t = reshape(matShape); // Reshape as rank 2 tensor. Broadcastable check made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooMatrixOld(matShape, t.entries.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public TensorOld toDense() {
        double[] entries = new double[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++) {
            entries[shape.entriesIndex(indices[i])] = this.entries[i];
        }

        return new TensorOld(shape, entries);
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specificing the full shape,
     * non-zero entries, and non-zero indices.</p>
     *
     * @return A human-readable string specificing the full shape, non-zero entries, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxEntries = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centring = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("Non-zero Entries: " + PrettyPrint.abbreviatedArray(entries, maxEntries, padding, precision, centring) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxEntries, padding, 20, centring));

        return sb.toString();
    }
}
