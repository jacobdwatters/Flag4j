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
import com.flag4j.core.ComplexTensorExclusiveMixin;
import com.flag4j.core.TensorExclusiveMixin;


/**
 * Complex sparse tensor.
 */
public class SparseCTensor extends ComplexSparseTensorBase
        implements TensorExclusiveMixin<SparseCTensor, CTensor, SparseCTensor, SparseCTensor>,
        ComplexTensorExclusiveMixin<CTensor> {


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


//    /**
//     * Create a sparse tensor from the dense entries of another tensor. This is not the recommended method for
//     * constructing sparse tensors. Use {@link #SparseCTensor(Shape, double[], int[][])} if the coordinate information
//     * is already known.
//     * @param shape Shape of the sparse tensor.
//     * @param entries Dense entries from which to construct sparse tensor.
//     */
//    public SparseCTensor(Shape shape, double[] entries) {
//        super(shape);
//
//        if(entries.length != shape.totalEntries()) {
//            throw new IllegalArgumentException(
//                    ErrorMessages.shapeEntriesError(shape, entries.length)
//            );
//        }
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(entries.length/2);
//        ArrayList<int[]> indices = new ArrayList<>(entries.length/2);
//        int[] entryIndices = new int[super.getRank()];
//
//        int prod;
//
//        for(int i=0; i<entries.length; i++) {
//
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(entryIndices.copy());
//            }
//
//            // Compute the indices for next entry based on the shape of the tensor.
//            prod = super.shape.totalEntries();
//            for(int j=0; j<entryIndices.length; j++) {
//                prod/=super.shape.dims[j];
//                if((i+1)%prod==0) {
//                    entryIndices[j] = (entryIndices[j]+1) % super.shape.dims[j];
//                }
//            }
//        }
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//        super.indices = new int[super.nonZeroEntries()][super.getRank()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//            super.indices[i] = indices.get(i);
//        }
//    }
//
//
//    /**
//     * Create a sparse tensor from the dense entries of another tensor. This is not the recommended method for
//     * constructing sparse tensors. Use {@link #SparseCTensor(Shape, double[], int[][])} if the coordinate information
//     * is already known.
//     * @param shape Shape of the sparse tensor.
//     * @param entries Dense entries from which to construct sparse tensor.
//     */
//    public SparseCTensor(Shape shape, int[] entries) {
//        super(shape);
//
//        if(entries.length != shape.totalEntries()) {
//            throw new IllegalArgumentException(
//                    ErrorMessages.shapeEntriesError(shape, entries.length)
//            );
//        }
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(entries.length/2);
//        ArrayList<int[]> indices = new ArrayList<>(entries.length/2);
//        int[] entryIndices = new int[super.getRank()];
//
//        int prod;
//
//        for(int i=0; i<entries.length; i++) {
//
//            if(entries[i]!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(entryIndices.copy());
//            }
//
//            // Compute the indices for next entry based on the shape of the tensor.
//            prod = super.shape.totalEntries();
//            for(int j=0; j<entryIndices.length; j++) {
//                prod/=super.shape.dims[j];
//                if((i+1)%prod==0) {
//                    entryIndices[j] = (entryIndices[j]+1) % super.shape.dims[j];
//                }
//            }
//        }
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//        super.indices = new int[super.nonZeroEntries()][super.getRank()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//            super.indices[i] = indices.get(i);
//        }
//    }
//
//
//    /**
//     * Create a sparse tensor from the dense entries of another tensor. This is not the recommended method for
//     * constructing sparse tensors. Use {@link #SparseCTensor(Shape, double[], int[][])} if the coordinate information
//     * is already known.
//     * @param shape Shape of the sparse tensor.
//     * @param entries Dense entries from which to construct sparse tensor.
//     */
//    public SparseCTensor(Shape shape, CNumber[] entries) {
//        super(shape);
//
//        if(entries.length != shape.totalEntries()) {
//            throw new IllegalArgumentException(
//                    ErrorMessages.shapeEntriesError(shape, entries.length)
//            );
//        }
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(entries.length/2);
//        ArrayList<int[]> indices = new ArrayList<>(entries.length/2);
//        int[] entryIndices = new int[super.getRank()];
//
//        int prod;
//
//        for(int i=0; i<entries.length; i++) {
//
//            if(entries[i].re!=0 || entries[i].im!=0) {
//                nonZeroEntries.add(entries[i]);
//                indices.add(entryIndices.copy());
//            }
//
//            // Compute the indices for next entry based on the shape of the tensor.
//            prod = super.shape.totalEntries();
//            for(int j=0; j<entryIndices.length; j++) {
//                prod/=super.shape.dims[j];
//                if((i+1)%prod==0) {
//                    entryIndices[j] = (entryIndices[j]+1) % super.shape.dims[j];
//                }
//            }
//        }
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//        super.indices = new int[super.nonZeroEntries()][super.getRank()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = nonZeroEntries.get(i).copy();
//            super.indices[i] = indices.get(i);
//        }
//    }


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
        return null;
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
        return null;
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
        return null;
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, interchanges the axes of this tensor so that it matches
     * the specified axes permutation and takes the complex conjugate of the elements of these axes. Same as {@link #hermTranspose(int[])}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     *             {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     * @throws IllegalArgumentException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     */
    @Override
    public CTensor H(int... axes) {
        return null;
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
    public SparseCTensor tensorDot(SparseCTensor src2, int aAxis, int bAxis) {
        return null;
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
    public SparseCTensor tensorDot(SparseCTensor src2, int[] aAxes, int[] bAxes) {
        return null;
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
    public SparseCTensor dot(SparseCTensor src2) {
        return null;
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
    public SparseCTensor transpose(int axis1, int axis2) {
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
     * @return The transpose of this tensor.
     */
    @Override
    public SparseCTensor T(int axis1, int axis2) {
        return null;
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
    public SparseCTensor transpose(int... axes) {
        return null;
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
    public SparseCTensor T(int... axes) {
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
    public SparseCTensor add(SparseTensor B) {
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
    public CTensor add(CTensor B) {
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
    public SparseCTensor sub(SparseTensor B) {
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
    public CTensor sub(CTensor B) {
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(SparseTensor B) {

    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void subEq(SparseTensor B) {

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
        return null;
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemMult(CTensor B) {
        return null;
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    @Override
    public CTensor elemDiv(CTensor B) {
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
    public SparseCTensor add(CNumber a) {
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
    public CTensor sub(SparseCTensor B) {
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
    public SparseCTensor sub(CNumber a) {
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank and stores the result in this tensor.
     *
     * @param B Second tensor in the subtraction.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public void addEq(SparseCTensor B) {

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
    public void subEq(SparseCTensor B) {

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
    public SparseCTensor abs() {
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
    public SparseCTensor elemDiv(SparseCTensor B) {
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
}
