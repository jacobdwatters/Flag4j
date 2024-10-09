/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.PrimitiveDoubleTensorBase;
import org.flag4j.arrays.backend.SparseTensorMixin;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.dense.real.AggregateDenseReal;
import org.flag4j.linalg.operations.sparse.coo.SparseDataWrapper;
import org.flag4j.linalg.operations.sparse.coo.real.RealCooTensorDot;
import org.flag4j.linalg.operations.sparse.coo.real.RealCooTensorOperations;
import org.flag4j.linalg.operations.sparse.coo.real.RealSparseEquals;
import org.flag4j.linalg.operations.sparse.coo.real_complex.RealComplexCooTensorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * <p>A real sparse tensor stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * primitive doubles.</p>
 *
 * <p>The {@link #entries non-zero entries} and {@link #indices non-zero indices} of a COO tensor are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse tensors allow for the efficient storage of and operations on tensors that contain many zero values.</p>
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).</p>
 *
 * <p>A sparse COO tensor is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #entries} of the tensor. All other entries in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #entries}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many operations assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.</p>
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero entries in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code entries[i]}.</p>
 *     </li>
 * </ul>
 */
public class CooTensor extends PrimitiveDoubleTensorBase<CooTensor, Tensor>
        implements SparseTensorMixin<Tensor, CooTensor> {

    /**
     * The non-zero indices of this tensor. Must have shape {@code (nnz, rank)}.
     */
    public final int[][] indices;
    /**
     * The number of non-zero entries in this tensor.
     */
    public final int nnz;
    /**
     * The sparsity of this matrix.
     */
    private double sparsity = -1;

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor. If this tensor is dense, this specifies all entries within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @param indices
     */
    public CooTensor(Shape shape, double[] entries, int[][] indices) {
        super(shape, entries);
        ValidateParameters.ensureArrayLengthsEq(entries.length, indices.length);
        if(indices.length != 0) ValidateParameters.ensureArrayLengthsEq(getRank(), indices[0].length);
        ValidateParameters.ensureTrue(shape.totalEntries().compareTo(BigInteger.valueOf(entries.length)) >= 0,
                "Tensor with shape " + shape + " cannot store " + entries.length + " entries.");
        this.indices = indices;
        this.nnz = entries.length;
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor. If this tensor is dense, this specifies all entries within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @param indices
     */
    public CooTensor(Shape shape, List<Double> entries, List<int[]> indices) {
        super(shape, ArrayUtils.fromDoubleList(entries));
        ValidateParameters.ensureArrayLengthsEq(entries.size(), indices.size());
        if(indices.size() != 0)ValidateParameters.ensureArrayLengthsEq(getRank(), indices.get(0).length);
        ValidateParameters.ensureTrue(shape.totalEntries().compareTo(BigInteger.valueOf(entries.size())) >= 0,
                "Tensor with shape " + shape + " cannot store " + entries.size() + " entries.");
        this.indices = indices.toArray(new int[0][]);
        this.nnz = super.entries.length;
    }


    /**
     * Creates a zero matrix with the specified shape.
     * @param shape The shape of the zero matrix to construct.
     */
    public CooTensor(Shape shape) {
        super(shape, new double[0]);
        this.indices = new int[0][getRank()];
        this.nnz = super.entries.length;
    }


    /**
     * Creates a sparse COO matrix with the specified shape, non-zero entries, and indices.
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero entries of the sparse COO matrix.
     * @param indices Indices of the non-zero entries in the sparse COO matrix.
     */
    public CooTensor(Shape shape, int[] entries, int[][] indices) {
        super(shape, ArrayUtils.asDouble(entries, null));
        ValidateParameters.ensureArrayLengthsEq(entries.length, indices.length);
        if(indices.length != 0) ValidateParameters.ensureArrayLengthsEq(getRank(), indices[0].length);
        ValidateParameters.ensureTrue(shape.totalEntries().compareTo(BigInteger.valueOf(entries.length)) >= 0,
                "Tensor with shape " + shape + " cannot store " + entries.length + " entries.");

        this.indices = indices;
        this.nnz = entries.length;
    }


    /**
     * Constructs a copy of the specified matrix.
     * @param b Matrix to make copy of.
     */
    public CooTensor(CooTensor b) {
        super(b.shape, b.entries.clone());
        this.indices = ArrayUtils.deepCopy(b.indices, null);
        this.nnz = b.nnz;
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Entries of the spares tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     */
    @Override
    public CooTensor makeLikeTensor(Shape shape, double[] entries) {
        return new CooTensor(shape, entries, ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    public CooTensor makeLikeTensor(Shape shape, double[] entries, int[][] indices) {
        return new CooTensor(shape, entries, indices);
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    public CooTensor makeLikeTensor(Shape shape, List<Double> entries, List<int[]> indices) {
        return new CooTensor(shape, entries, indices);
    }


    /**
     * Makes a dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     *
     * @return A dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     */
    public Tensor makeDenseTensor(Shape shape, double[] entries) {
        return new Tensor(shape, entries);
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
     * @return The density of this sparse tensor.
     */
    @Override
    public double sparsity() {
        // Check if the sparsity has already been computed.
        if (this.sparsity < 0) {
            BigInteger totalEntries = totalEntries();
            BigDecimal sparsity = new BigDecimal(totalEntries).subtract(BigDecimal.valueOf(nnz));
            sparsity = sparsity.divide(new BigDecimal(totalEntries), 50, RoundingMode.HALF_UP);

            this.sparsity = sparsity.doubleValue();
        }


        return sparsity;
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    @Override
    public Tensor toDense() {
        double[] entries = new double[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++)
            entries[shape.entriesIndex(indices[i])] = this.entries[i];

        return new Tensor(shape, entries);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public Double get(int... indices) {
        ValidateParameters.ensureValidIndex(shape, indices);
        if(entries.length == 0) return null; // Can not get reference of field so no way to get zero element.

        for(int i=0; i<nnz; i++)
            if(Arrays.equals(this.indices[i], indices)) return entries[i];

        return 0.0; // Return zero if the index is not found.
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param Index Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooTensor set(Double value, int... index) {
        ValidateParameters.ensureValidIndex(shape, index);
        CooTensor dest;

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
            dest = new CooTensor(shape, entries.clone(), ArrayUtils.deepCopy(indices, null));
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

            dest = new CooTensor(shape, newEntries, newIndices);
            dest.sortIndices();
        }

        return dest;
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CooTensor flatten() {
        int[][] destIndices = new int[entries.length][1];

        for(int i=0, size=entries.length; i<size; i++)
            destIndices[i][0] = shape.entriesIndex(indices[i]);

        return makeLikeTensor(new Shape(shape.totalEntries().intValueExact()), entries.clone(), destIndices);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public CooTensor flatten(int axis) {
        ValidateParameters.ensureIndexInBounds(indices[0].length, axis);
        int[][] destIndices = new int[indices.length][indices[0].length];

        // Compute new shape.
        int[] destShape = new int[indices[0].length];
        Arrays.fill(destShape, 1);
        destShape[axis] = shape.totalEntries().intValueExact();

        for(int i=0, size=entries.length; i<size; i++)
            destIndices[i][axis] = shape.entriesIndex(indices[i]);

        return makeLikeTensor(new Shape(destShape), entries.clone(), destIndices);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public CooTensor reshape(Shape newShape) {
        ValidateParameters.ensureBroadcastable(shape, newShape);

        int rank = indices[0].length;
        int newRank = newShape.getRank();
        int nnz = entries.length;

        int[] oldStrides = shape.getStrides();
        int[] newStrides = newShape.getStrides();

        int[][] newIndices = new int[nnz][newRank];

        for(int i=0; i<nnz; i++) {
            int[] idxRow = indices[i];
            int[] newIdxRow = newIndices[i];

            int flatIndex = 0;
            for(int j=0; j < rank; j++) {
                flatIndex += idxRow[j] * oldStrides[j];
            }

            for(int j=0; j<newRank; j++) {
                newIdxRow[j] = flatIndex / newStrides[j];
                flatIndex %= newStrides[j];
            }
        }

        return makeLikeTensor(newShape, entries.clone(), newIndices);
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public CooTensor sub(Double b) {
        return super.sub(b);  // Overrides superclass to emphasize this method only acts on non-zero entries of the tensor.
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Double b) {
        super.subEq(b); // Overrides superclass to emphasize this method only acts on non-zero entries of the tensor.
    }


    /**
     * Adds a scalar field value to each non-zero entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public CooTensor add(Double b) {
        return super.add(b); // Overrides superclass to emphasize this method only acts on non-zero entries of the tensor.
    }


    /**
     * Adds a scalar value to each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        super.addEq(b); // Overrides superclass to emphasize this method only acts on non-zero entries of the tensor.
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooTensor add(CooTensor b) {
        return RealCooTensorOperations.add(this, b);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCTensor add(CooCTensor b) {
        return RealComplexCooTensorOperations.add(b, this);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooTensor sub(CooTensor b) {
        return RealCooTensorOperations.sub(this, b);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooTensor elemMult(CooTensor b) {
        return RealCooTensorOperations.elemMult(this, b);
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
    public Tensor tensorDot(CooTensor src2, int[] aAxes, int[] bAxes) {
        return RealCooTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}. This will be a tensor of rank
     * {@code this.getRank() - 2} with the same shape as this tensor but with {@code axis1} and {@code axis2} removed.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public CooTensor tensorTr(int axis1, int axis2) {
        // Validate parameters.
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.ensureValidIndices(getRank(), axis1, axis2);
        ValidateParameters.ensureEquals(shape.get(axis1), shape.get(axis2));

        int rank = getRank();
        int[] dims = shape.getDims();

        // Determine the shape of the resulting tensor.
        int[] traceShape = new int[rank - 2];
        int newShapeIndex = 0;
        for (int i = 0; i < rank; i++) {
            if (i != axis1 && i != axis2) {
                traceShape[newShapeIndex++] = dims[i];
            }
        }

        // Use a map to accumulate non-zero entries that are on the diagonal.
        Map<Integer, Double> resultMap = new HashMap<>();
        int[] strides = shape.getStrides();

        // Iterate through the non-zero entries and accumulate trace for those on the diagonal.
        for (int i = 0; i < this.nnz; i++) {
            int[] indices = this.indices[i];
            double value = this.entries[i];

            // Check if the current entry is on the diagonal
            if (indices[axis1] == indices[axis2]) {
                // Compute a linear index for the resulting tensor by ignoring axis1 and axis2.
                int linearIndex = 0;
                int stride = 1;

                for (int j = rank - 1; j >= 0; j--) {
                    if (j != axis1 && j != axis2) {
                        linearIndex += indices[j] * stride;
                        stride *= dims[j];
                    }
                }

                // Accumulate the value in the result map.
                resultMap.put(linearIndex, resultMap.getOrDefault(linearIndex, 0.0) + value);
            }
        }

        // Construct the result tensor from the accumulated non-zero entries
        int resultNnz = resultMap.size();
        int[][] resultIndices = new int[resultNnz][rank - 2];
        double[] resultEntries = new double[resultNnz];
        int resultIndex = 0;

        for (Map.Entry<Integer, Double> entry : resultMap.entrySet()) {
            int linearIndex = entry.getKey();
            double entryValue = entry.getValue();

            // Use the getIndices method to convert the flat index to n-dimensional index.
            int[] multiDimIndices = shape.getIndices(linearIndex);

            // Copy relevant dimensions to resultIndices, excluding axis1 and axis2.
            int resultDimIndex = 0;
            for (int j = 0; j < rank; j++) {
                if (j != axis1 && j != axis2) {
                    resultIndices[resultIndex][resultDimIndex++] = multiDimIndices[j];
                }
            }

            resultEntries[resultIndex] = entryValue;
            resultIndex++;
        }

        return makeLikeTensor(new Shape(traceShape), resultEntries, resultIndices);
    }


    /**
     * <p>Computes the product of all non-zero values in this tensor.</p>
     *
     * <p>NOTE: This is <b>only</b> the product of the non-zero values in this tensor.</p>
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public Double prod() {
        // Override from FieldTensorBase to emphasize that the product is only for non-zero entries.
        return super.prod();
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public CooTensor T(int axis1, int axis2) {
        int rank = getRank();
        ValidateParameters.ensureIndexInBounds(rank, axis1, axis2);

        if(axis1 == axis2) return copy(); // Simply return a copy.

        int[][] transposeIndices = new int[nnz][rank];
        double[] transposeEntries = new double[nnz];

        for(int i=0; i<nnz; i++) {
            transposeEntries[i] = entries[i];
            transposeIndices[i] = indices[i].clone();
            ArrayUtils.swap(transposeIndices[i], axis1, axis2);
        }

        // Create sparse coo tensor and sort values lexicographically by indices.
        CooTensor transpose = makeLikeTensor(shape.swapAxes(axis1, axis2), transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public CooTensor T(int... axes) {
        int rank = getRank();
        ValidateParameters.ensureEquals(rank, axes.length);
        ValidateParameters.ensurePermutation(axes);

        int[][] transposeIndices = new int[nnz][rank];
        double[] transposeEntries = new double[nnz];

        // Permute the indices according to the permutation array.
        for(int i=0; i < nnz; i++) {
            transposeEntries[i] = entries[i];
            transposeIndices[i] = indices[i].clone();

            for(int j = 0; j < rank; j++) {
                transposeIndices[i][j] = indices[i][axes[j]];
            }
        }

        // Create sparse COO tensor and sort values lexicographically by indices.
        CooTensor transpose = makeLikeTensor(shape.swapAxes(axes), transposeEntries, transposeIndices);
        transpose.sortIndices();

        return transpose;
    }


    /**
     * <p>Computes the element-wise reciprocals of the non-zero elements of this sparse tensor.</p>
     *
     * <p>Note: This method <b>only</b> computes the reciprocals of the non-zero elements.</p>
     *
     * @return A tensor containing the reciprocal non-zero elements of this tensor.
     */
    @Override
    public CooTensor recip() {
        /* This method is override from FieldTensorBase to make clear it is only computing the
            multiplicative inverse for the non-zero elements of the tensor */
        double[] recip = new double[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            recip[i] = 1.0/entries[i];

        return makeLikeTensor(shape, recip);
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public CooTensor copy() {
        return new CooTensor(shape, entries.clone(), ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Finds the minimum non-zero value in this tensor. If this tensor is complex, then this method finds the smallest value in
     * magnitude.
     *
     * @return The minimum non-zero value (smallest in magnitude for a complex valued tensor) in this tensor. If this tensor does
     * not have any non-zero values, then {@code null} will be returned.
     */
    @Override
    public double min() {
        // Overrides method in super class to emphasize that the method works on the non-zero elements only.
        return super.min();
    }


    /**
     * Finds the maximum non-zero value in this tensor. If this tensor is complex, then this method finds the largest value in
     * magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor. If this tensor does not have any
     * non-zero values, then {@code null} will be returned.
     */
    @Override
    public double max() {
        // Overrides method in super class to emphasize that the method works on the non-zero elements only.
        return super.max();
    }


    /**
     * Finds the minimum non-zero value, in absolute value, in this tensor.
     *
     * @return The minimum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        // Overrides method in super class to emphasize that the method works on the non-zero elements only.
        return super.minAbs();
    }


    /**
     * Finds the maximum non-zero value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        // Overrides method in super class to emphasize that the method works on the non-zero elements only.
        return super.maxAbs();
    }


    /**
     * Finds the indices of the minimum non-zero value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values, then an empty array is returned.
     */
    @Override
    public int[] argmin() {
        if(nnz > 0) return indices[AggregateDenseReal.argmin(entries)];
        else return new int[0];
    }


    /**
     * Finds the indices of the maximum non-zero value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values, then an empty array is returned.
     */
    @Override
    public int[] argmax() {
        if(nnz > 0) return indices[AggregateDenseReal.argmin(entries)];
        else return new int[0];
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values, then an empty array is returned.
     */
    @Override
    public int[] argminAbs() {
        if(nnz > 0) return indices[AggregateDenseReal.argminAbs(entries)];
        else return new int[0];
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned. If this tensor has no non-zero values, then an empty array is returned.
     */
    @Override
    public int[] argmaxAbs() {
        if(nnz > 0) return indices[AggregateDenseReal.argminAbs(entries)];
        else return new int[0];
    }


    /**
     * Adds a scalar value to each non-zero value of this tensor.
     *
     * @param b Value to add to each non-zero value of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public CooTensor add(double b) {
        // Overrides method in super class to emphasize that the method works on the non-zero elements only.
        return super.add(b);
    }


    /**
     * Subtracts a scalar value from each non-zero value of this tensor.
     *
     * @param b Value to subtract from each non-zero value of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public CooTensor sub(double b) {
        // Overrides method in super class to emphasize that the method works on the non-zero elements only.
        return super.sub(b);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooTensor src2 = (CooTensor) object;

        return RealSparseEquals.cooTensorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (entries[i] != 0.0) {
                result = 31*result + Double.hashCode(entries[i]);
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * Converts this real COO tensor to an equivalent complex COO tensor.
     * @return A complex COO tensor equivalent to this tensor.
     */
    public CooCTensor toComplex() {
        return new CooCTensor(shape, ArrayUtils.wrapAsComplex128(entries, null), ArrayUtils.deepCopy(indices, null));
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specifying the full shape,
     * non-zero entries, and non-zero indices.</p>
     *
     * @return A human-readable string specifying the full shape, non-zero entries, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centring = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("Non-zero Entries: " + PrettyPrint.abbreviatedArray(entries, maxCols, padding, precision, centring) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centring));

        return sb.toString();
    }
}
