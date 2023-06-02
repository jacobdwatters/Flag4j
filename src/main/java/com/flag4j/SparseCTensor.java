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
import com.flag4j.core.sparse.ComplexSparseTensorBase;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.operations.sparse.complex.ComplexSparseEquals;
import com.flag4j.operations.sparse.real_complex.RealComplexSparseEquals;


/**
 * Complex sparse tensor.
 */
public class SparseCTensor
        extends ComplexSparseTensorBase<SparseCTensor, CTensor, SparseTensor>
//        implements ComplexTensorExclusiveMixin<SparseCTensor>
{


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public SparseCTensor(Shape shape) {
        super(shape, 0, new CNumber[0], new int[0][0]);
    }


    /**
     * Creates a sparse tensor with specified shape filled with zeros.
     * @param shape Shape of the tensor.
     * @param nonZeroEntries Non-zero entries of the tensor.
     * @param indices Indices of the non-zero entries of the tensor.
     */
    public SparseCTensor(Shape shape, double[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], indices);

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
    public SparseCTensor(Shape shape, int[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], indices);

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
    public SparseCTensor(Shape shape, CNumber[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, indices);
    }


    /**
     * Constructs a sparse complex tensor whose non-zero values, indices, and shape are specified by another sparse complex
     * tensor.
     * @param A The sparse complex tensor to construct a copy of.
     */
    public SparseCTensor(SparseCTensor A) {
        super(A.shape.copy(), A.nonZeroEntries(), A.entries.clone(), new int[A.indices.length][A.indices[0].length]);
        for(int i=0; i<indices.length; i++) {
            super.indices[i] = A.indices[i].clone();
        }
    }


    /**
     * Checks if an object is equal to this sparse tensor object. Valid object types are: {@link Tensor}, {@link CTensor},
     * {@link SparseTensor}, and {@link SparseCTensor}. These tensors are equal to this tensor if all entries are
     * numerically equal to the corresponding element of this tensor.
     *
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors are numerically equivalent and false otherwise.
     */
    @Override
    public boolean equals(Object object) {
        boolean equal = false;

        if(object instanceof Tensor) {
            Tensor tensor = (Tensor) object;
            equal = RealComplexDenseSparseEquals.tensorEquals(tensor, this);

        } else if(object instanceof CTensor) {
            CTensor tensor = (CTensor) object;
            equal = ComplexDenseSparseEquals.tensorEquals(tensor, this);

        } else if(object instanceof SparseTensor) {
            SparseTensor tensor = (SparseTensor) object;
            equal = RealComplexSparseEquals.tensorEquals(tensor, this);

        } else if(object instanceof SparseCTensor) {
            SparseCTensor tensor = (SparseCTensor) object;
            equal = ComplexSparseEquals.tensorEquals(tensor, this);

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected SparseCTensor getSelf() {
        return this;
    }


    /**
     * Checks if this tensor has only real valued entries.
     *
     * @return True if this tensor contains <b>NO</b> complex entries. Otherwise, returns false.
     */
    @Override
    public boolean isReal() {
        return false;
    }


    /**
     * Checks if this tensor contains at least one complex entry.
     *
     * @return True if this tensor contains at least one complex entry. Otherwise, returns false.
     */
    @Override
    public boolean isComplex() {
        return false;
    }


    /**
     * Computes the complex conjugate of a tensor.
     *
     * @return The complex conjugate of this tensor.
     */
    @Override
    public SparseCTensor conj() {
        return null;
    }


    /**
     * Converts a complex tensor to a real matrix. The imaginary component of any complex value will be ignored.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     */
    @Override
    public SparseTensor toReal() {
        return null;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public SparseCTensor hermTranspose() {
        return null;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public SparseCTensor H() {
        return null;
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
    public SparseCTensor set(CNumber value, int... indices) {
        return null;
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return false;
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return false;
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
    public SparseCTensor reshape(int... shape) {
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
    public SparseCTensor set(double value, int... indices) {
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
    public SparseCTensor reshape(Shape shape) {
        return null;
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public SparseCTensor flatten() {
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
    public SparseCTensor add(SparseCTensor B) {
        return null;
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CTensor add(double a) {
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
    public SparseCTensor sub(SparseCTensor B) {
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
        return null;
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCTensor mult(double factor) {
        return null;
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCTensor mult(CNumber factor) {
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
    public SparseCTensor div(double divisor) {
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
    public SparseCTensor div(CNumber divisor) {
        return null;
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public CNumber sum() {
        return null;
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public SparseCTensor sqrt() {
        return null;
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public SparseTensor abs() {
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCTensor transpose() {
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCTensor T() {
        return null;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public SparseCTensor recip() {
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
    public CNumber get(int... indices) {
        return null;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public SparseCTensor copy() {
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
    public SparseCTensor elemMult(SparseCTensor B) {
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
    public SparseCTensor elemDiv(CTensor B) {
        return null;
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public double min() {
        return 0;
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public double max() {
        return 0;
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return 0;
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return 0;
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
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return 0;
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
        return 0;
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double infNorm() {
        return 0;
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public SparseCTensor flatten(int axis) {
        return null;
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sparseSort() {

    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public CTensor toDense() {
        CNumber[] entries = new CNumber[totalEntries().intValueExact()];

        for(int i=0; i<nonZeroEntries; i++) {
            entries[shape.entriesIndex(indices[i])] = this.entries[i];
        }

        return new CTensor(shape.copy(), entries);
    }
}
