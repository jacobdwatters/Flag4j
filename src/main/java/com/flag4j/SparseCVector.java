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
import com.flag4j.core.ComplexSparseTensorBase;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.util.ArrayUtils;

/**
 * Complex sparse vector.
 */
public class SparseCVector
        extends ComplexSparseTensorBase<SparseCVector, CVector, SparseVector>
//        implements VectorMixin<SparseCVector, CVector, SparseCVector, SparseCVector, CNumber, SparseCMatrix, CMatrix, SparseCMatrix>
{


    /**
     * The size of this vector. That is, the number of entries in this vector.
     */
    public final int size;
    /**
     * Indices of non-zero entries in this sparse vector.
     */
    public final int[] indices;


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     */
    public SparseCVector(int size) {
        super(new Shape(size), 0, new CNumber[0], new int[0][0]);
        this.size = size;
        this.indices = new int[0];
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, int[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices})
        );
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
        this.size = size;
        this.indices = indices;
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, double[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                new CNumber[nonZeroEntries.length],
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices}
        ));
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
        this.size = size;
        this.indices = indices;
    }


    /**
     * Creates a sparse column vector of specified size filled with zeros.
     * @param size The size of the sparse vector. i.e. the total number of entries in the sparse vector.
     * @param nonZeroEntries The nonZero entries of this sparse vector.
     * @param indices Indices of the nonZero entries.
     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
     * the length of the nonZeroEntries array is greater than the size.
     */
    public SparseCVector(int size, CNumber[] nonZeroEntries, int[] indices) {
        super(new Shape(size),
                nonZeroEntries.length,
                nonZeroEntries,
                RealDenseTranspose.blockedIntMatrix(new int[][]{indices})
        );
        this.size = size;
        this.indices = indices;
    }


    /**
     * Constructs a complex sparse vector whose size, orientation, non-zero entries, and indices are specified
     * by another complex sparse vector.
     * @param a Vector to copy.
     */
    public SparseCVector(SparseCVector a) {
        super(a.shape.copy(),
                a.nonZeroEntries(),
                new CNumber[a.nonZeroEntries()],
                RealDenseTranspose.blockedIntMatrix(new int[][]{a.indices})
        );
        ArrayUtils.copy2CNumber(a.entries, super.entries);
        this.size = a.size;
        this.indices = a.indices.clone();
    }


//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(int[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(double[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(CNumber[] entries) {
//        super(entries.length, VectorOrientations.COL);
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i].re!=0 && entries[i].im!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = nonZeroEntries.get(i).copy();
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


    // TODO: These methods (for all sparse tensor classes) will be moved into factory methods
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(int[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(double[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse column vector from a dense array.
//     * @param entries Dense entries of the vector.
//     * @param orientation Orientation of the vector.
//     * @throws IllegalArgumentException If the lengths of nonZeroEntries and indices arrays are not equal or if
//     * the length of the nonZeroEntries array is greater than the size.
//     */
//    public SparseCVector(CNumber[] entries, VectorOrientations orientation) {
//        super(entries.length, orientation);
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(super.totalEntries()/8);
//        ArrayList<Integer> indices = new ArrayList<>(super.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            if(entries[i].re!=0 && entries[i].im!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(i);
//            }
//        }
//
//        super.entries = new CNumber[super.nonZeroEntries()];
//        for(int i=0; i<super.entries.length; i++) {
//            super.entries[i] = nonZeroEntries.get(i).copy();
//        }
//        super.indices = indices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected SparseCVector getSelf() {
        return this;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public SparseCVector hermTranspose() {
        return null;
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public SparseCVector H() {
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
    public SparseCVector set(CNumber value, int... indices) {
        return null;
    }

    @Override
    public boolean isReal() {
        return false;
    }

    @Override
    public boolean isComplex() {
        return false;
    }

    @Override
    public SparseCVector conj() {
        return null;
    }

    @Override
    public SparseVector toReal() {
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
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public SparseCVector set(double value, int... indices) {
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
    public SparseCVector reshape(Shape shape) {
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
    public SparseCVector reshape(int... shape) {
        return null;
    }

    /**
     * Flattens tensor to single dimension. To flatten tensor along a single axis.
     *
     * @return The flattened tensor.
     */
    @Override
    public SparseCVector flatten() {
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
    public SparseCVector add(SparseCVector B) {
        return null;
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(double a) {
        return null;
    }

    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(CNumber a) {
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
    public SparseCVector sub(SparseCVector B) {
        return null;
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector sub(double a) {
        return null;
    }

    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVector sub(CNumber a) {
        return null;
    }

    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(SparseCVector B) {

    }

    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(CNumber b) {

    }

    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void addEq(Double b) {

    }

    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(SparseCVector B) {

    }

    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(CNumber b) {

    }

    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    @Override
    public void subEq(Double b) {

    }

    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCVector mult(double factor) {
        return null;
    }

    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public SparseCVector mult(CNumber factor) {
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
    public SparseCVector div(double divisor) {
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
    public SparseCVector div(CNumber divisor) {
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
    public SparseCVector sqrt() {
        return null;
    }

    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public SparseVector abs() {
        return null;
    }

    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCVector transpose() {
        return null;
    }

    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCVector T() {
        return null;
    }

    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public SparseCVector recip() {
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
    public SparseCVector copy() {
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
    public SparseCVector elemMult(SparseCVector B) {
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
    public SparseCVector elemDiv(CVector B) {
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
    public SparseCVector flatten(int axis) {
        return null;
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sparseSort() {

    }


    /**
     * Gets the full size of this vector (including non-zero entries).
     * @return The full size of this vector.
     */
    public int size() {
        return shape.totalEntries().intValueExact();
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public CVector toDense() {
        CNumber[] entries = new CNumber[size];

        for(int i=0; i<nonZeroEntries; i++) {
            entries[indices[i]] = this.entries[i];
        }

        return new CVector(entries);
    }
}
