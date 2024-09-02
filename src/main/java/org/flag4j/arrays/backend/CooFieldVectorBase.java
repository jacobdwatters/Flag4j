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

package org.flag4j.arrays.backend;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.fields.RealFloat64;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooFieldVector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.operations.sparse.coo.field_ops.CooFieldVectorOperations;
import org.flag4j.operations_old.sparse.coo.SparseDataWrapper;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;
import java.util.List;


/**
 * <p>A sparse vector stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
 * elements of a {@link Field}.</p>
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
 *
 * @param <T> Type of this vector.
 * @param <U> Type of matrix equivalent to {@code T}.
 * @param <V> Type of equivalent dense vector.
 * @param <W> Type of dense matrix equivalent to {@code U}.
 * @param <Y> Type of the field element in this vector.
 */
public abstract class CooFieldVectorBase<T extends CooFieldVectorBase<T, U, V, W, Y>, U extends CooFieldMatrixBase<U, W, T, Y>,
        V extends DenseFieldVectorBase<V, W, T, Y>, W extends DenseFieldMatrixBase<W, U, ?, V, Y>, Y extends Field<Y>>
        extends FieldTensorBase<T, V, Y>
        implements SparseVectorMixin<T, V, U, W, Y> {

    /**
     * Indices of the non-zero values of this sparse COO vector.
     */
    public final int[] indices;
    /**
     * The number of non-zero entries in this sparse COO vector.
     */
    public final int nnz;
    /**
     * The total size of this sparse COO vector (including zero values).
     */
    public final int size;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected CooFieldVectorBase(int size, Y[] entries, int[] indices) {
        super(new Shape(size), entries);
        if(entries.length != indices.length) {
            throw new IllegalArgumentException("entries and indices arrays of a COO vector must have the same length but got " +
                    "lengths" + entries.length + " and " + indices.length + ".");
        }
        if(entries.length > size) {
            throw new IllegalArgumentException("The number of entries cannot be greater than the size of the vector but but got " +
                    "entries.length=" + entries.length + " and size=" + size + ".");
        }

        this.indices = indices;
        this.size = size;
        this.nnz = entries.length;
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and non-zero indices.
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     * @return A sparse COO vector of the same type as this vector with the specified {@code size}, non-zero entries,
     * and non-zero indices.
     */
    public abstract T makeLikeTensor(int size, Y[] entries, int[] indices);


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and the same
     * non-zero indices as this vector.
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @return A sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and the same
     * non-zero indices as this vector.
     */
    public abstract T makeLikeTensor(int size, Y[] entries);


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and non-zero indices.
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     * @return A sparse COO vector of the same type as this vector with the specified {@code size}, non-zero entries,
     * and non-zero indices.
     */
    public abstract T makeLikeTensor(int size, List<Y> entries, List<Integer> indices);


    /**
     * Constructs a dense vector which is of a similar type to this sparse COO vector containing the specified {@code entries}.
     * @param entries The entries of the dense vector.
     * @return A dense vector which is of a similar type to this sparse COO vector containing the specified {@code entries}.
     */
    public abstract V makeLikeDenseTensor(Y... entries);


    /**
     * Constructs a sparse matrix which is of a similar type to this sparse COO vector with the specified {@code shape}, non-zero
     * entries, non-zero row indices, and non-zero column indices.
     * @param shape Shape of the matrix.
     * @param entries The non-zero indices of the matrix.
     * @param rowIndices The row indices of the non-zero entries.
     * @param colIndices The column indices of the non-zero entries.
     * @return A dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     */
    public abstract U makeLikeMatrix(Shape shape, Y[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Constructs a dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     * @param shape Shape of the dense matrix.
     * @param entries The entries of the dense matrix.
     * @return A dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     */
    public abstract W makeLikeDenseMatrix(Shape shape, Y... entries);



    /**
     * Constructs a sparse COO vector of the specified size filled with zeros.
     * @param size The size of the vector to construct.
     * @return A sparse COO vector of the specified size filled with zeros.
     */
    public abstract T makeZeroVector(int size);


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
    public T H(int axis1, int axis2) {
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
        return conj();
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
    public T H(int... axes) {
        ParameterChecks.ensurePermutation(axes);
        return conj();
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
    public V tensorDot(T src2, int[] aAxes, int[] bAxes) {
        if(aAxes.length != 1 || bAxes.length != 1) {
            throw new LinearAlgebraException("Vector dot product requires exactly one dimension for each vector but got "
                    + aAxes.length + " and " + bAxes.length + ".");
        }
        if(aAxes[0] != 0 || bAxes[0] != 0) {
            throw new LinearAlgebraException("Both axes must be 0 for vector dot product but got "
                    + aAxes[0] + " and " + bAxes[0] + ".");
        }

        return makeLikeDenseTensor(dot(src2));
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
    public T T(int axis1, int axis2) {
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
    public T T(int... axes) {
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
    public T join(T b) {
        Y zero = getZeroElement();
        if(zero == null) zero = b.getZeroElement();

        Field<Y>[] newEntries = new Field[this.entries.length + b.entries.length];
        Arrays.fill(newEntries, zero);
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

        return makeLikeTensor(this.size + b.size, (Y[]) newEntries, newIndices);
    }


    /**
     * <p>Computes the inner product between two vectors.</p>
     * 
     * <p>Note: this method is distinct from {@link #dot(CooFieldVectorBase)}. The inner product is equivalent to the dot product 
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #dot(CooFieldVectorBase) 
     */
    @Override
    public Y inner(T b) {
        return CooFieldVectorOperations.inner(this, b);
    }


    /**
     * <p>Computes the dot product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #inner(CooFieldVectorBase)}. The inner product is equivalent to the dot product 
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #inner(CooFieldVectorBase) 
     */
    @Override
    public Y dot(T b) {
        return CooFieldVectorOperations.dot(this, b);
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
    public T normalize() {
        if(entries.length == 0 || nnz == 0) return makeZeroVector(size); // Return early for no non-zero values.
        double norm = VectorNorms.norm(entries);
        return norm==0 ? makeZeroVector(size) : this.div(norm);
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     *
     * @see #isPerp(CooFieldVectorBase) 
     */
    @Override
    public boolean isParallel(T b) {
        final double tol = 1.0e-12; // Tolerance to accommodate floating point arithmetic error in scaling.
        boolean result;

        if(this.size!=b.size) {
            return false;
        } else if(this.size<=1) {
            return true;
        } else if(this.isZeros() || b.isZeros()) {
            return true; // Any vector is parallel to zero vector.
        } else {
            result = true;
            int sparseIndex = 0;
            Y scale = getZeroElement();
            if(scale == null) scale = b.getZeroElement();

            // Find first non-zero entry in b and compute the scaling factor (we know there is at least one from else-if).
            for(int i=0; i<b.size; i++) {
                if(!b.entries[i].isZero()) {
                    scale = this.entries[i].div(b.entries[this.indices[i]]);
                    break;
                }
            }

            for(int i=0; i<b.size; i++) {
                if(sparseIndex >= this.entries.length || i!=this.indices[sparseIndex]) {
                    // Then this index is not in the sparse vector.
                    if(!b.entries[i].isZero()) return false;
                } else {
                    // Ensure the scaled entry is approximately equal to the entry in this vector.
                    if(this.entries[sparseIndex].sub(scale.mult(b.entries[i])).mag() > tol) return false;
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
     * @see #isParallel(CooFieldVectorBase)
     */
    @Override
    public boolean isPerp(T b) {
        return this.size!=b.size ? false : inner(b).isZero();
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector (including zeros).
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
    public V toDense() {
        Field<Y>[] denseEntries = new Field[size];
        Arrays.fill(denseEntries, nnz > 0 ? denseEntries[0].getZero() : null);

        for(int i = 0; i < nnz; i++)
            denseEntries[indices[i]] = entries[i];

        return makeLikeDenseTensor((Y[]) denseEntries);
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
    public Y get(int... indices) {
        ParameterChecks.ensureEquals(indices.length, 1);
        ParameterChecks.ensureInRange(indices[0], 0, size, "index");
        Y zero = getZeroElement();

        int idx = Arrays.binarySearch(this.indices, indices[0]);
        return idx>=0 ? entries[idx] : zero;
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
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
    public T flatten(int axis) {
        ParameterChecks.ensureValidAxes(shape, axis);
        return copy();
    }


    /**
     * Computes the product of all non-zero values in this tensor.
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public Y prod() {
        return super.prod(); // Overrides method from super class to emphasize it operates on the non-zero values only.
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
    public T add(T b) {
        return CooFieldVectorOperations.add((T) this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with the scalar {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        return (T) CooFieldVectorOperations.sub(this, b);
    }


    /**
     * Adds a real value to each entry non-zero value of this tensor.
     *
     * @param b Value to add to each non-zero value of this tensor.
     *
     * @return Sum of this tensor with {@code b}.
     */
    @Override
    public T add(double b) {
        return super.add(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Subtracts a real value from each non-zero entry of this tensor.
     *
     * @param b Value to subtract from each non-zero value of this tensor.
     *
     * @return Difference of this tensor with {@code b}.
     */
    @Override
    public T sub(double b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
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
    public T elemMult(T b) {
        return (T) CooFieldVectorOperations.elemMult(this, b);
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
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        throw new LinearAlgebraException("Tensor trace cannot be computed for a rank 1 tensor " +
                "(must be rank 2 or " + "greater).");
    }


    /**
     * Adds a scalar field value to each non-zero entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor's non-zero with the scalar {@code b}.
     */
    @Override
    public T add(Y b) {
        return super.add(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Adds a scalar value to each non-zero entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(Y b) {
        super.addEq(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Subtracts a scalar field value from each non-zero value of this tensor.
     *
     * @param b Scalar field value in difference.
     *
     * @return The difference of this tensor's non-zero values and the scalar {@code b}.
     */
    @Override
    public T sub(Y b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Subtracts a scalar value from each non-zero value of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(Y b) {
        super.subEq(b); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Finds the minimum non-zero value in this tensor.
     *
     * @return The minimum non-zero value in this tensor.
     */
    @Override
    public Y min() {
        return super.min(); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Finds the maximum non-zero value in this tensor.
     *
     * @return The maximum non-zero value in this tensor.
     */
    @Override
    public Y max() {
        return super.max(); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Finds the minimum non-zero value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return super.minAbs(); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Finds the maximum non-zero value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum non-zero value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return super.maxAbs(); // Overrides method from super class to emphasize it operates on the non-zero values only.
    }


    /**
     * Finds the indices of the minimum non-zero value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        return new int[]{indices[CompareField.argmin(entries)]};
    }


    /**
     * Finds the indices of the maximum non-zero value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return new int[]{indices[CompareField.argmax(entries)]};
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return new int[]{indices[CompareField.argminAbs(entries)]};
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return new int[]{indices[CompareField.argmaxAbs(entries)]};
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
    public T T() {
        return copy();
    }


    /**
     * Computes the Hermitian transpose of a tensor by exchanging and conjugating the first and last axes of this tensor.
     *
     * @return The Hermitian transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public T H() {
        return copy();
    }


    /**
     * Computes the element-wise reciprocals of this tensor's non-zero values.
     *
     * @return A tensor containing the reciprocal elements of this tensor's non-zero values.
     */
    @Override
    public T recip() {
        return super.recip();  // Overrides method from super to emphasize that it works only on the non-zero values of this vector.
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public CooFieldVector<RealFloat64> abs() {
        RealFloat64[] abs = new RealFloat64[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            abs[i] = new RealFloat64(entries[i].abs());

        return new CooFieldVector<RealFloat64>(size, abs, indices.clone());
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
    public U stack(T b) {
        return (U) CooFieldVectorOperations.stack(this, b);
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
    public U repeat(int n, int axis) {
        return (U) CooFieldVectorOperations.repeat(this, n, axis);
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
    public U stack(T b, int axis) {
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
    public W outer(T b) {
        return (W) CooFieldVectorOperations.outerProduct(this, b);
    }
}
