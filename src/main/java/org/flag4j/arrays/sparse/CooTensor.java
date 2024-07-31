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
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.core.TensorExclusiveMixin;
import org.flag4j.core.sparse_base.RealSparseTensorBase;
import org.flag4j.operations.dense.real.RealDenseOperations;
import org.flag4j.operations.sparse.coo.real.RealSparseEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.List;

/**
 * Real sparse tensor. Can be any rank. Stored in coordinate (COO) format.
 */
public class CooTensor
        extends RealSparseTensorBase<CooTensor, Tensor, CooCTensor, CTensor>
        implements TensorExclusiveMixin<CooTensor, Tensor, CooTensor, CooCTensor>
{


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public CooTensor(Shape shape) {
        super(shape, 0, new double[0], new int[0][0]);
        this.shape.makeStridesIfNull();
    }


    /**
     * Creates a sparse tensor with specified shape and non-zero values/indices.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooTensor(Shape shape, double[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, indices);
        this.shape.makeStridesIfNull();
    }


    /**
     * Creates a sparse tensor with specified shape and non-zero values/indices.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public CooTensor(Shape shape, int[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, ArrayUtils.asDouble(nonZeroEntries, null), indices);
        this.shape.makeStridesIfNull();
    }


    /**
     * Constructs a sparse tensor whose shape and non-zero values/indices are given by another sparse tensor.
     * This effectively copies the tensor.
     * @param A Tensor to copy.
     */
    public CooTensor(CooTensor A) {
        super(A.shape, A.nonZeroEntries(), A.entries.clone(), new int[A.indices.length][A.indices[0].length]);
        shape.makeStridesIfNull();
        for(int i=0; i<indices.length; i++) {
            super.indices[i] = A.indices[i].clone();
        }
    }


    /**
     * Checks if an object is equal to this sparse COO tensor.
     * @param object Object to compare this sparse COO tensor to.
     * @return True if the object is a {@link CooTensor}, has the same shape as this tensor, and is element-wise equal to this
     * tensor.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooTensor src2 = (CooTensor) object;
        return RealSparseEquals.tensorEquals(this, src2);
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CooCTensor toComplex() {
        return new CooCTensor(
                shape,
                ArrayUtils.copy2CNumber(entries, null),
                ArrayUtils.deepCopy(indices, null)
        );
    }


    /**
     * Converts a sparse {@link CooTensor} from a dense {@link Tensor}. This is likely only worthwhile for very sparse tensors.
     * @param src Dense tensor to convert to sparse COO tensor.
     * @return A COO tensor which is equivalent to the {@code src} dense tensor.
     */
    public static CooTensor fromDense(Tensor src) {
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

        return new CooTensor(src.shape, ArrayUtils.fromDoubleList(entries), indices.toArray(new int[0][]));
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
    protected CooTensor makeTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooTensor(shape, entries, indices);
    }


    /**
     * A factory for creating a real dense tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected Tensor makeDenseTensor(Shape shape, double[] entries) {
        return new Tensor(shape, entries);
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
    protected CooCTensor makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
    }

    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public CooTensor set(double value, int... indices) {
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
    public CooTensor reshape(Shape newShape) {
        ParameterChecks.assertBroadcastable(shape, newShape);
        newShape.makeStridesIfNull(); // Ensure this shape object has strides computed.

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

        return new CooTensor(newShape, entries.clone(), newIndices);
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis see {@link #flatten(int)}.
     *
     * @return The flattened tensor.
     * @see #flatten(int)
     */
    @Override
    public CooTensor flatten() {
        int[][] destIndices = new int[entries.length][1];

        for(int i = 0; i < entries.length; i++)
            destIndices[i][0] = RealDenseOperations.prod(indices[i]);

        return new CooTensor(shape, entries.clone(), destIndices);
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
    public CooTensor tensorDot(CooTensor src2, int[] aAxes, int[] bAxes) {
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
    public CooTensor tensorDot(CooTensor src2) {
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
    public CooTensor T(int axis1, int axis2) {
        // TODO: Implementation.
        return null;
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
    public CooTensor T(int... axes) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooTensor add(CooTensor B) {
        // TODO: Implementation.
        return null;
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
    public Tensor add(Tensor B) {
        // TODO: Implementation.
        return null;
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
    public Tensor sub(Tensor B) {
        // TODO: Implementation.
        return null;
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
        // TODO: Implementation.
        return null;
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
    public CooCTensor add(CooCTensor B) {
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
    public Tensor add(double a) {
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
    public CTensor add(CNumber a) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CooTensor sub(CooTensor B) {
        // TODO: Implementation.
        return null;
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
        // TODO: Implementation.
        return null;
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
    public CooCTensor sub(CooCTensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(CooTensor B) {

    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     *
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(CooTensor B) {

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
    public CooTensor elemMult(Tensor B) {
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
    public Tensor sub(double a) {
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
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CooTensor mult(double factor) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CooCTensor mult(CNumber factor) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CooTensor div(double divisor) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CooCTensor div(CNumber divisor) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public CooTensor sqrt() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public CooTensor abs() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooTensor transpose() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooTensor T() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public CooTensor recip() {
        // TODO: Implementation.
        return null;
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
        // TODO: Implementation.
        return null;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CooTensor copy() {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CooTensor elemMult(CooTensor B) {
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
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
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
     *
     * @return The result of the element-wise tensor division.
     *
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemDiv(CTensor B) {
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
    public CooTensor elemDiv(Tensor B) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link #tensorDot(TensorBase, int) this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product operation.
     * A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     *
     * @return The 'inverse' of this tensor as defined in the above sense.
     *
     * @see #tensorInv()
     */
    @Override
    public CooTensor tensorInv(int numIndices) {
        // TODO: Implementation.
        return null;
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
    public CooTensor reshape(int... shape) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return new int[0];
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return new int[0];
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CooTensor flatten(int axis) {
        // TODO: Implementation
        return null;
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CooTensor getSelf() {
        return this;
    }


    @Override
    public boolean allClose(CooTensor tensor, double relTol, double absTol) {
        return RealSparseEquals.allCloseTensor(this, tensor, relTol, absTol);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public Tensor toDense() {
        double[] entries = new double[totalEntries().intValueExact()];

        for(int i=0; i<nonZeroEntries; i++) {
            entries[shape.entriesIndex(indices[i])] = this.entries[i];
        }

        return new Tensor(shape, entries);
    }
}
