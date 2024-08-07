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
import org.flag4j.core.sparse_base.ComplexSparseTensorBase;
import org.flag4j.operations.sparse.coo.complex.ComplexSparseEquals;

import java.util.ArrayList;
import java.util.List;


/**
 * Complex sparse tensor. Stored in coordinate (COO) format.
 */
public class CooCTensor
        extends ComplexSparseTensorBase<CooCTensor, CTensor, CooTensor>
//        implements ComplexTensorExclusiveMixin<CooCTensor>  // TODO: Implement methods from this class.
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
    public CooCTensor(Shape shape, CNumber[] nonZeroEntries, int[][] indices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, indices);
        this.shape.makeStridesIfNull();
    }


    /**
     * Constructs a sparse complex tensor whose non-zero values, indices, and shape are specified by another sparse complex
     * tensor.
     * @param A The sparse complex tensor to construct a copy of.
     */
    public CooCTensor(CooCTensor A) {
        super(A.shape.copy(), A.nonZeroEntries(), A.entries.clone(), new int[A.indices.length][A.indices[0].length]);
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
    public CooCTensor conj() {
        return null;
    }


    /**
     * Converts a complex tensor to a real matrix. The imaginary component of any complex value will be ignored.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     */
    @Override
    public CooTensor toReal() {
        return null;
    }


    /**
     * Converts a complex tensor to a real matrix safely. That is, first checks if the tensor only contains real values
     * and then converts to a real tensor. However, if non-real value exist, then an error is thrown.
     *
     * @return A tensor of the same size containing only the real components of this tensor.
     * @throws RuntimeException If this tensor contains at least one non-real value.
     * @see #toReal()
     */
    @Override
    public CooTensor toRealSafe() {
        return null;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CooCTensor hermTranspose() {
        return null;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CooCTensor H() {
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
    public CooCTensor set(CNumber value, int... indices) {
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
    public CooCTensor reshape(int... shape) {
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
    public CooCTensor reshape(Shape shape) {
        return null;
    }


    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public CooCTensor flatten() {
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
    public CooCTensor add(CooCTensor B) {
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
    public CooCTensor sub(CooCTensor B) {
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
    public CooCTensor mult(double factor) {
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
    public CooCTensor div(double divisor) {
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
    public CooCTensor sqrt() {
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
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooCTensor transpose() {
        return null;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CooCTensor T() {
        return null;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public CooCTensor recip() {
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
    public CooCTensor copy() {
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
    public CooCTensor elemMult(CooCTensor B) {
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
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     */
    @Override
    public CooCTensor flatten(int axis) {
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

        return new CooCTensor(src.shape.copy(), entries.toArray(new CNumber[0]), indices.toArray(new int[0][]));
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
