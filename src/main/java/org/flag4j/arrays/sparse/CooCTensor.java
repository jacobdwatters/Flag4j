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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.ComplexTensorExclusiveMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.core.sparse_base.ComplexSparseTensorBase;
import org.flag4j.operations.dense.real.RealDenseOperations;
import org.flag4j.operations.dense.real.RealDenseTranspose;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;
import org.flag4j.operations.sparse.coo.complex.ComplexCooTensorOperations;
import org.flag4j.operations.sparse.coo.complex.ComplexSparseEquals;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexCooTensorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Complex sparse tensor. Stored in coordinate (COO) format.
 */
public class CooCTensor
        extends ComplexSparseTensorBase<CooCTensor, CTensor, CooTensor>
        implements ComplexTensorExclusiveMixin<CooCTensor>
{


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public CooCTensor(Shape shape) {
        super(shape, 0, new CNumber[0], new int[0][0]);
        this.shape.makeStridesIfNull();
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooCTensor(Shape shape, double[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], indices);
        this.shape.makeStridesIfNull();

        for(int i=0; i<indices.length; i++) {
            super.entries[i] = new CNumber(nonZeroEntries[i]);
        }
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooCTensor(Shape shape, int[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], indices);
        this.shape.makeStridesIfNull();

        for(int i=0; i<nnz; i++) {
            super.entries[i] = new CNumber(nonZeroEntries[i]);
        }
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooCTensor(Shape shape, CNumber[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, indices);
        this.shape.makeStridesIfNull();
    }


    /**
     * Creates a sparse tensor with specified shape and non-zero values/indices.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooCTensor(Shape shape, List<CNumber> nonZeroEntries, List<int[]> indices) {
        super(shape, nonZeroEntries.size(),
                nonZeroEntries.toArray(new CNumber[0]),
                indices.toArray(new int[0][]));
        this.shape.makeStridesIfNull();
    }


    /**
     * Constructs a sparse complex tensor whose non-zero values, indices, and shape are specified by another sparse complex
     * tensor.
     * @param A The sparse complex tensor to construct a copy of.
     */
    public CooCTensor(CooCTensor A) {
        super(A.shape, A.nonZeroEntries(), A.entries.clone(), new int[A.indices.length][A.indices[0].length]);
        shape.makeStridesIfNull();

        for(int i=0; i<indices.length; i++) {
            super.indices[i] = A.indices[i].clone();
        }
    }


    /**
     * Checks if an object is equal to this sparse COO tensor.
     * @param object Object to compare this sparse COO tensor to.
     * @return True if the object is a {@link CooCTensor}, has the same shape as this tensor, and is element-wise equal to this
     * tensor.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooCTensor src2 = (CooCTensor) object;
        return ComplexSparseEquals.tensorEquals(this, src2);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CooCTensor getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooCTensor tensor, double relTol, double absTol) {
        return ComplexSparseEquals.allCloseTensor(this, tensor, relTol, absTol);
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CooCTensor hermTranspose() {
        return H(0, getRank()-1);
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CooCTensor H() {
        return H(0, getRank()-1);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this matrix for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException  If the number of indices is not equal to the rank of this tensor.
     * @throws IndexOutOfBoundsException If any of the indices are not within this tensor.
     */
    @Override
    public CooCTensor set(CNumber value, int... indices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public CooCTensor set(double value, int... indices) {
        // TODO: Implementation.
        return null;
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
    public CooCTensor reshape(Shape newShape) {
        ParameterChecks.assertBroadcastable(shape, newShape);
        newShape.makeStridesIfNull();

        int rank = indices[0].length;
        int nnz = entries.length;

        int[] oldStrides = shape.getStrides();
        int[] newStrides = newShape.getStrides();

        int[][] newIndices = new int[nnz][rank];

        for (int i = 0; i < nnz; i++) {
            int flatIndex = 0;
            for (int j = 0; j < rank; j++) {
                flatIndex += indices[i][j] * oldStrides[j];
            }

            for (int j = 0; j < rank; j++) {
                newIndices[i][j] = flatIndex / newStrides[j];
                flatIndex %= newStrides[j];
            }
        }

        return new CooCTensor(newShape, ArrayUtils.copyOf(entries), newIndices);
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public CooCTensor flatten() {
        int[][] destIndices = new int[entries.length][1];

        for(int i = 0; i < entries.length; i++)
            destIndices[i][0] = RealDenseOperations.prod(indices[i]);

        return new CooCTensor(shape, ArrayUtils.copyOf(entries), destIndices);
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 Tensor to contract with this tensor.
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
    public CooCTensor tensorDot(CooCTensor src2, int[] aAxes, int[] bAxes) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     *
     * @param src2 Tensor to compute dot product with this tensor.
     *
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     *
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     *                                  along the second-to-last axis.
     */
    @Override
    public CooCTensor tensorDot(CooCTensor src2) {
        // TODO: Implementation.
        return null;
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
    public CooCTensor T(int axis1, int axis2) {
        int rank = getRank();
        ParameterChecks.assertIndexInBounds(rank, axis1, axis2);

        if(axis1 == axis2) return copy(); // Simply return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        CNumber[] transposeEntries = new CNumber[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i].copy();
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create sparse coo tensor and sort values lexicographically.
        CooCTensor transpose = new CooCTensor(shape, transposeEntries, transposeIndices);
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
    public CooCTensor T(int... axes) {
        int rank = getRank();
        ParameterChecks.assertEquals(rank, axes.length);
        ParameterChecks.assertPermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        CNumber[] transposeEntries = new CNumber[nnz];

        // Permute the indices according to the permutation array.
        for(int i = 0; i < nnz; i++) {
            transposeEntries[i] = entries[i].copy();
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create sparse coo tensor and sort values lexicographically.
        CooCTensor transpose = new CooCTensor(shape, transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
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
    public CooCTensor add(CooTensor B) {
        return RealComplexCooTensorOperations.add(this, B);
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
    public CTensor add(Tensor B) {
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
    public CTensor sub(Tensor B) {
        return RealComplexDenseSparseOperations.sub(this, B);
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
    public CTensor add(CTensor B) {
        return ComplexDenseSparseOperations.add(B, this);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooCTensor add(CooCTensor B) {
        return ComplexCooTensorOperations.add(this, B);
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
    public CooCTensor sub(CooTensor B) {
        return RealComplexCooTensorOperations.sub(this, B);
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
    public CTensor sub(CTensor B) {
        return ComplexDenseSparseOperations.sub(this, B);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CTensor add(double a) {
        return RealComplexDenseSparseOperations.add(this, a);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CTensor add(CNumber a) {
        return ComplexDenseSparseOperations.add(this, a);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooCTensor sub(CooCTensor B) {
        return ComplexCooTensorOperations.sub(this, B);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensor elemMult(Tensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensor elemMult(CooTensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemMult(CTensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CTensor sub(double a) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CTensor sub(CNumber a) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooCTensor transpose() {
        return T(0, getRank()-1);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooCTensor T() {
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
    public CNumber get(int... indices) {
        ParameterChecks.assertEquals(indices.length, getRank());

        for(int i = 0; i < nnz; i++) {
            if(Arrays.equals(this.indices[i], indices)) {
                return entries[i];
            }
        }

        return CNumber.zero(); // Return zero if the index is not found
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CooCTensor copy() {
        return new CooCTensor(shape, ArrayUtils.copyOf(entries), ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooCTensor elemMult(CooCTensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooCTensor elemDiv(CTensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide from this tensor.
     *
     * @return The result of the element-wise tensor division.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CooCTensor elemDiv(Tensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link #tensorDot(TensorBase, int)  this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product operation.
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
    public CTensor tensorInv(int numIndices) {
        return toDense().tensorInv(numIndices);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CooCTensor flatten(int axis) {
        // TODO: Implementation.
        return null;
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CooCTensor makeTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
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
    protected CooTensor makeRealTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooTensor(shape, entries, indices);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
        // TODO: Implementation
    }


    /**
     * Converts a sparse {@link CooCTensor} from a dense {@link Tensor}. This is likely only worthwhile for very sparse tensors.
     * @param src Dense tensor to convert to sparse COO tensor.
     * @return A COO tensor which is equivalent to the {@code src} dense tensor.
     */
    public static CooCTensor fromDense(CTensor src) {
        List<CNumber> entries = new ArrayList<>();
        List<int[]> indices = new ArrayList<>();

        int size = src.entries.length;
        CNumber value;

        for(int i=0; i<size; i++) {
            value = src.entries[i].copy();

            if(value.equals(CNumber.zero())) {
                entries.add(value);
                indices.add(src.shape.getIndices(i));
            }
        }

        return new CooCTensor(src.shape, entries.toArray(new CNumber[0]), indices.toArray(new int[0][]));
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be broadcastable with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     */
    public CooCMatrix toMatrix(Shape matShape) {
        ParameterChecks.assertRank(2, matShape);
        CooCTensor t = reshape(matShape); // Reshape as rank 2 tensor. Broadcastable check made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooCMatrix(matShape, t.entries.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public CTensor toDense() {
        CNumber[] entries = new CNumber[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++) {
            entries[shape.entriesIndex(indices[i])] = this.entries[i];
        }

        return new CTensor(shape, entries);
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
     *
     * @return The conjugate transpose of this tensor.
     */
    @Override
    public CooCTensor H(int axis1, int axis2) {
        int rank = getRank();
        ParameterChecks.assertIndexInBounds(rank, axis1, axis2);

        if(axis1 == axis2) return copy(); // Simply return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        CNumber[] transposeEntries = new CNumber[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i].conj();
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create sparse coo tensor and sort values lexicographically.
        CooCTensor transpose = new CooCTensor(shape, transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation and takes the complex conjugate of the elements of these axes. Same as {@link #hermTranspose(int[])}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public CooCTensor H(int... axes) {
        int rank = getRank();
        ParameterChecks.assertEquals(rank, axes.length);
        ParameterChecks.assertPermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        CNumber[] transposeEntries = new CNumber[nnz];

        // Permute the indices according to the permutation array.
        for(int i = 0; i < nnz; i++) {
            transposeEntries[i] = entries[i].conj();
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create sparse coo tensor and sort values lexicographically.
        CooCTensor transpose = new CooCTensor(shape, transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     *
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    @Override
    public void addEq(Tensor B) {
        // TODO: Remove. It does not make sense for this to be in a sparse tensor.
    }


    /**
     * Computes the element-wise addition of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the addition.
     *
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    @Override
    public void addEq(CooCTensor B) {
        // TODO: Remove. It does not make sense for this to be in a sparse tensor.
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     *
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    @Override
    public void subEq(Tensor B) {
        // TODO: Remove. It does not make sense for this to be in a sparse tensor.
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     *
     * @throws IllegalArgumentException If this tensor and {@code B} have different shapes.
     */
    @Override
    public void subEq(CooCTensor B) {
        // TODO: Remove. It does not make sense for this to be in a sparse tensor.FEq
    }
}
