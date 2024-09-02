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

import org.flag4j.arrays.backend.PrimitiveDoubleTensorBase;
import org.flag4j.arrays.backend.SparseVectorMixin;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.Shape;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations.dense.real.AggregateDenseReal;
import org.flag4j.operations.sparse.coo.SparseDataWrapper;
import org.flag4j.operations.sparse.coo.real.RealCooVectorOperations;
import org.flag4j.operations.sparse.coo.real.RealSparseEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;
import java.util.List;


/**
 * <p>A real sparse vector stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
 * primitive doubles.</p>
 *
 * <p>The {@link #entries non-zero entries} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse vectors allow for the efficient storage of and operations on vectors that contain many zero values.</p>
 *
 * <p>COO vectors are optimized for hyper-sparse vectors (i.e. vectors which contain almost all zeros relative to the size of the
 * vector).</p>
 *
 * <p>A sparse COO vector is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #entries} of the vector. All other entries in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.</p>
 */
public class CooVector extends PrimitiveDoubleTensorBase<CooVector, Vector>
        implements SparseVectorMixin<CooVector, Vector, CooMatrix, Matrix, Double> {

    /**
     * The indices of the non-zero entries in this sparse COO vector.
     */
    public final int[] indices;
    /**
     * The full size of this sparse COO vector (including the zeros.).
     */
    public final int size;
    /**
     * The number of non-zero values in this sparse COO vector.
     */
    public final int nnz;


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero entries of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(Shape shape, double[] entries, int[] indices) {
        super(shape, entries);
        ParameterChecks.ensureRank(shape, 1);
        this.size = shape.get(0);
        this.indices = indices;
        this.nnz = entries.length;
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero entries of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(int size, double[] entries, int[] indices) {
        super(new Shape(size), entries);
        this.size = size;
        this.indices = indices;
        this.nnz = entries.length;
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero entries of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooVector(int size, List<Double> entries, List<Integer> indices) {
        super(new Shape(size), ArrayUtils.fromDoubleList(entries));
        this.indices = ArrayUtils.fromIntegerList(indices);
        this.size = size;
        this.nnz = this.entries.length;
    }


    /**
     * Creates a zero vector of the specified {@code size}.
     */
    public CooVector(int size) {
        super(new Shape(size), new double[0]);
        indices = new int[0];
        nnz = 0;
        this.size = size;
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooVector makeLikeTensor(Shape shape, double[] entries) {
        return new CooVector(shape, entries, indices.clone());
    }


    /**
     * Constructs a vector of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero entries of the vector to construct.
     * @param indices Indices of the non-zero values in this vector.
     *
     * @return A vector of the same type as this tensor with the given the shape and entries.
     */
    public CooVector makeLikeTensor(int size, double[] entries, int[] indices) {
        return new CooVector(shape, entries, indices);
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
    public Vector tensorDot(CooVector src2, int[] aAxes, int[] bAxes) {
        if(aAxes.length != 1 || bAxes.length != 1) {
            throw new LinearAlgebraException("Vector dot product requires exactly one dimension for each vector but got "
                    + aAxes.length + " and " + bAxes.length + ".");
        }
        if(aAxes[0] != 0 || bAxes[0] != 0) {
            throw new LinearAlgebraException("Both axes must be 0 for vector dot product but got "
                    + aAxes[0] + " and " + bAxes[0] + ".");
        }

        return new Vector(dot(src2));
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
    public CooVector T(int axis1, int axis2) {
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
        return copy();
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
    public CooVector T(int... axes) {
        ParameterChecks.ensurePermutation(axes);
        return copy();
    }


    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public CooVector join(CooVector b) {
        double[] newEntries = new double[this.entries.length + b.entries.length];
        Arrays.fill(newEntries, 0d);
        int[] newIndices = new int[this.indices.length + b.indices.length];

        // Copy values from this vector.
        System.arraycopy(this.entries, 0, newEntries, 0, this.entries.length);
        // Copy values from vector b.
        System.arraycopy(b.entries, 0, newEntries, this.entries.length, b.entries.length);

        // Copy indices from this vector.
        System.arraycopy(this.indices, 0, newIndices, 0, this.entries.length);

        // Copy the indices from vector b with a shift.
        for(int i=0; i<b.indices.length; i++) {
            newIndices[this.indices.length+i] = b.indices[i] + this.size;
        }

        return makeLikeTensor(this.size + b.size, newEntries, newIndices);
    }


    /**
     * <p>Computes the inner product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #dot(CooVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #dot(VectorMixin)
     */
    @Override
    public Double inner(CooVector b) {
        return RealCooVectorOperations.inner(this, b);
    }


    /**
     * <p>Computes the dot product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #inner(CooVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #inner(CooVector) 
     */
    @Override
    public Double dot(CooVector b) {
        return inner(b);
    }


    /**
     * Computes the Euclidean norm of this vector.
     *
     * @return The Euclidean norm of this vector.
     */
    @Override
    public double norm() {
        return VectorNorms.norm(entries);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    @Override
    public double norm(int p) {
        return VectorNorms.norm(entries, p);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public CooVector normalize() {
        double norm = VectorNorms.norm(entries);
        return norm==0 ? new CooVector(size) : this.div(norm);
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     *
     * @see #isPerp(CooVector) 
     */
    @Override
    public boolean isParallel(CooVector b) {
        final double tol = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        boolean result;

        if(this.size!=b.size) {
            return false;
        } else if(this.size<=1) {
            return true;
        } else if(this.isZeros() || b.isZeros()) {
            return true; // Any vector is parallel to a zero vector.
        } else {
            result = true;
            int sparseIndex = 0;
            double scale = 0;

            // Find first non-zero entry in b and compute the scaling factor (we know there is at least one from else-if).
            for(int i=0; i<b.size; i++) {
                if(b.entries[i]!=0) {
                    scale = this.entries[i]/b.entries[this.indices[i]];
                    break;
                }
            }

            for(int i=0; i<b.size; i++) {
                if(sparseIndex >= this.entries.length || i!=this.indices[sparseIndex]) {
                    // Then this index is not in the sparse vector.
                    if(b.entries[i] != 0) return false;

                } else {
                    // Ensure the scaled entry is approximately equal to the entry in this vector.
                    if(Math.abs(b.entries[i]*scale - this.entries[sparseIndex]) > tol) return false;
                    sparseIndex++;
                }
            }
        }

        return true;
    }


    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     *
     * @see #isParallel(CooVector)
     * @implNote This method checks if the vector is perpendicular by checking if the inner product is essentially zero:
     * {@code Math.abs(this.inner(b)) < TOL} where {@code TOL} is a small non-negative value.
     */
    @Override
    public boolean isPerp(CooVector b) {
        final double TOL = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        return this.size!=b.size ? false : Math.abs(this.inner(b)) < TOL;
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return size;
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
     * @return The density of this sparse tensor.
     */
    @Override
    public double sparsity() {
        return 1.0 - ((double) nnz / (double) size);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    @Override
    public Vector toDense() {
        double[] entries = new double[size];
        for(int i = 0; i<nnz; i++)
            entries[indices[i]] = this.entries[i];

        return new Vector(entries);
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
        SparseDataWrapper.wrap(entries, indices).sparseSort().unwrap(entries, indices);
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
        ParameterChecks.ensureEquals(indices.length, 1);
        ParameterChecks.ensureInRange(indices[0], 0, size, "index");

        int idx = Arrays.binarySearch(this.indices, indices[0]);
        return idx>=0 ? entries[idx] : 0;
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CooVector flatten() {
        return copy();
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
    public CooVector flatten(int axis) {
        ParameterChecks.ensureValidAxes(shape, axis);
        return copy();
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public CooVector reshape(Shape newShape) {
        ParameterChecks.ensureRank(newShape, 1);
        ParameterChecks.ensureBroadcastable(shape, newShape);
        return copy();
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor.
     *
     * @param b Scalar value in difference.
     *
     * @return The difference of this tensor's non-zero values and the scalar {@code b}.
     */
    @Override
    public CooVector sub(Double b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Subtracts a scalar value from each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Double b) {
        super.subEq(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Adds a scalar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public CooVector add(Double b) {
        return super.add(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Double b) {
        super.addEq(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
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
    public CooVector add(CooVector b) {
        return RealCooVectorOperations.add(this, b);
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
    public CooVector sub(CooVector b) {
        return RealCooVectorOperations.sub(this, b);
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
    public CooVector elemMult(CooVector b) {
        return RealCooVectorOperations.elemMult(this, b);
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
    public CooVector tensorTr(int axis1, int axis2) {
        throw new LinearAlgebraException("Tensor trace cannot be computed for a rank 1 tensor " +
                "(must be rank 2 or " + "greater).");
    }


    /**
     * Computes the product of all non-zero values in this tensor.
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public Double prod() {
        return super.prod(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public CooVector T() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public CooVector H() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public CooVector H(int axis1, int axis2) {
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
        return copy();
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public CooVector H(int... axes) {
        ParameterChecks.ensurePermutation(axes);
        return copy();
    }


    /**
     * Computes the element-wise reciprocals of the non-zero values of this tensor.
     *
     * @return A tensor containing the reciprocal elements of the non-zero values of this tensor.
     */
    @Override
    public CooVector recip() {
        return super.recip(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Finds the minimum non-zero value in this tensor.
     *
     * @return The minimum non-zero value in this tensor.
     */
    @Override
    public Double min() {
        return super.min(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Finds the maximum non-zero value in this tensor.
     *
     * @return The maximum non-zero value in this tensor.
     */
    @Override
    public Double max() {
        return super.max(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Finds the minimum non-zero value, in absolute value, in this tensor.
     *
     * @return The minimum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return super.minAbs(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Finds the maximum non-zero value, in absolute value, in this tensor.
     *
     * @return The maximum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return super.maxAbs(); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Finds the indices of the minimum non-zero value in this tensor.
     *
     * @return The indices of the minimum non-zero value in this tensor.
     */
    @Override
    public int[] argmin() {
        return new int[]{indices[AggregateDenseReal.argmin(entries)]};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return new int[]{indices[AggregateDenseReal.argmax(entries)]};
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return new int[]{indices[AggregateDenseReal.argminAbs(entries)]};
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return new int[]{indices[AggregateDenseReal.argmaxAbs(entries)]};
    }


    /**
     * Adds a scalar value to each non-zero element of this tensor.
     *
     * @param b Value to add to each non-zero entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each entry of this tensor.
     */
    @Override
    public CooVector add(double b) {
        return super.add(b); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }


    /**
     * Subtracts a scalar value non-zero from each element of this tensor.
     *
     * @param b Value to subtract from each non-zero entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each entry of this tensor.
     */
    @Override
    public CooVector sub(double b) {
        return super.sub(b); // Overrides method from super class to emphasize that this method works on the non-zero values only.
    }



    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooVector src2 = (CooVector) object;

        return RealSparseEquals.cooVectorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (entries[i] != 0.0) {
                result = 31*result + Double.hashCode(entries[i]);
                result = 31*result + Integer.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    @Override
    public CooMatrix repeat(int n, int axis) {
        return RealCooVectorOperations.repeat(this, n, axis);
    }


    /**
     * Stacks two vectors vertically as if they were row vectors to form a matrix with two rows.
     *
     * @param b Vector to stack below this vector.
     *
     * @return The result of stacking this vector and vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public CooMatrix stack(CooVector b) {
        return RealCooVectorOperations.stack(this, b);
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CooMatrix stack(CooVector b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        return axis==0 ? stack(b) : stack(b).T();
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public Matrix outer(CooVector b) {
        return RealCooVectorOperations.outerProduct(this, b);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.</p>
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.</p>
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public CooMatrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new CooMatrix(this.size, 1, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new CooMatrix(1, this.size, entries.clone(), rowIndices, colIndices);
        }
    }
}
