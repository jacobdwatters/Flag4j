/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.arrays.backend.semiring_arrays;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseVectorData;
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.sparse.SparseValidation;
import org.flag4j.linalg.ops.common.semiring_ops.AggregateSemiring;
import org.flag4j.linalg.ops.sparse.SparseUtils;
import org.flag4j.linalg.ops.sparse.coo.CooConcat;
import org.flag4j.linalg.ops.sparse.coo.CooDataSorter;
import org.flag4j.linalg.ops.sparse.coo.CooGetSet;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringVectorOps;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;


/**
 * <p>A sparse vector stored in coordinate list (COO) format. The {@link #data} of this COO vector are
 * elements of a {@link Semiring}.
 *
 * <p>The {@link #data non-zero data} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero data is fixed.
 *
 * <p>Sparse vectors allow for the efficient storage of and ops on large vectors that contain many zero values.
 *
 * <p>COO vectors are optimized for large hyper-sparse vectors (i.e. vectors which contain almost all zeros relative to the size of the
 * vector).
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #data} of the vector. All other data in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the data of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.
 *
 * @param <T> Type of this vector.
 * @param <U> Type of equivalent dense vector.
 * @param <V> Type of matrix equivalent to {@code T}.
 * @param <Y> Type of dense matrix equivalent to {@code U}.
 * @param <Y> Type of the semiring element in this vector.
 */
public abstract class AbstractCooSemiringVector<
        T extends AbstractCooSemiringVector<T, U, V, W, Y>,
        U extends AbstractDenseSemiringVector<U, W, Y>,
        V extends AbstractCooSemiringMatrix<V, W, T, Y>,
        W extends AbstractDenseSemiringMatrix<W, U, Y>,
        Y extends Semiring<Y>>
        extends AbstractTensor<T, Y[], Y>
        implements SemiringTensorMixin<T, U, Y>, VectorMixin<T, V, W, Y> {

    /**
     * The zero element for the semiring that this tensor's elements belong to.
     */
    private Y zeroElement;
    /**
     * Indices of the non-zero values of this sparse COO vector.
     */
    public final int[] indices;
    /**
     * The number of non-zero data in this sparse COO vector.
     */
    public final int nnz;
    /**
     * The total size of this sparse COO vector (including zero values).
     */
    public final int size;
    /**
     * The sparsity of this matrix.
     */
    public final double sparsity;


    /**
     * Creates a sparse COO semiring vector with the specified data and shape.
     *
     * @param shape Shape of this vector. Must be rank-1.
     * @param data Non-zero data of this COO vector.
     * @param indices Non-zero indices of this COO vector.
     */
    protected AbstractCooSemiringVector(Shape shape, Y[] data, int[] indices) {
        super(shape, data);
        ValidateParameters.ensureRank(shape, 1);
        this.size = shape.get(0);
        SparseValidation.validateCoo(size, data.length, indices);

        this.indices = indices;
        this.nnz = data.length;
        sparsity = BigDecimal.valueOf(nnz).divide(new BigDecimal(shape.totalEntries()), RoundingMode.HALF_UP).doubleValue();

        // Attempt to set the zero element for the semiring.
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
    }


    /**
     * Creates a tensor with the specified data and shape without performing <em>any</em> validation on the parameters.
     *
     * @param shape Shape of this tensor.
     * @param data Non-zero entries of the tensor.
     * @param indices Non-zero entries of the tensor.
     * @param dummy Dummy object to distinguish this constructor from the safe variant.
     */
    protected AbstractCooSemiringVector(Shape shape, Y[] data, int[] indices, Object dummy) {
        super(shape, data);

        this.size = shape.get(0);
        this.indices = indices;
        this.nnz = data.length;
        sparsity = BigDecimal.valueOf(nnz).divide(new BigDecimal(shape.totalEntries()), RoundingMode.HALF_UP).doubleValue();

        // Attempt to set the zero element for the semiring.
        this.zeroElement = (data.length > 0 && data[0] != null) ? data[0].getZero() : null;
    }


    /**
     * Gets the size of the 1D data array backing this tensor.
     *
     * @return The size of the 1D data array backing this tensor.
     */
    @Override
    public int dataLength() {
        return data.length;
    }


    /**
     * Constructs a sparse COO vector of the same type as this vector with the specified non-zero data and indices.
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero data of the vector to construct.
     * @param indices Non-zero row indices of the vector to construct.
     * @return A sparse COO vector of the same type as this vector with the specified non-zero data and indices.
     */
    public abstract T makeLikeTensor(Shape shape, Y[] entries, int[] indices);


    /**
     * Constructs a dense vector of a similar type as this vector with the specified shape and data.
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the vector to construct.
     * @return A dense vector of a similar type as this vector with the specified data.
     */
    public abstract U makeLikeDenseTensor(Shape shape, Y... entries);


    /**
     * Constructs a dense matrix of a similar type as this vector with the specified shape and data.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     * @return A dense matrix of a similar type as this vector with the specified data.
     */
    public abstract W makeLikeDenseMatrix(Shape shape, Y... entries);


    /**
     * Constructs a COO vector with the specified shape, non-zero data, and non-zero indices.
     * @param shape Shape of the vector.
     * @param entries Non-zero values of the vector.
     * @param indices Indices of the non-zero values in the vector.
     * @return A COO vector of the same type as this vector with the specified shape, non-zero data, and non-zero indices.
     */
    public abstract T makeLikeTensor(Shape shape, List<Y> entries, List<Integer> indices);


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and row and column indices.
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     * @return A COO matrix of similar type as this vector with the specified shape, non-zero data, and non-zero row/col indices.
     */
    public abstract V makeLikeMatrix(Shape shape, Y[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Gets the sparsity of this matrix as a decimal percentage.
     * That is, the percentage of data in this matrix that are zero.
     * @return The sparsity of this matrix as a decimal percentage.
     * @see #density()
     */
    public double sparsity() {
        return sparsity;
    }


    /**
     * Gets the density of this matrix as a decimal percentage.
     * That is, the percentage of data in this matrix that are non-zero.
     * @return The density of this matrix as a decimal percentage.
     * @see #sparsity
     */
    public double density() {
        return 1.0 - sparsity;
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        CooDataSorter.wrap(data, indices).sparseSort().unwrap(data, indices);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param target Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws IndexOutOfBoundsException If any {target} are not within this tensor.
     */
    @Override
    public Y get(int... target) {
        ValidateParameters.ensureArrayLengthsEq(1, target.length);
        return get(target[0]);
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
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
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
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public T T(int... axes) {
        if(axes.length != 1)
            throw new IllegalArgumentException("Axes for tensor of rank 1 must be permutation of {1}.");
        ValidateParameters.ensurePermutation(axes);
        return copy();
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, data);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param target Indices of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(Y value, int... target) {
        ValidateParameters.validateTensorIndex(shape, target);
        int idx = Arrays.binarySearch(indices, target[0]);

        Y[] destEntries;
        int[] destIndices;

        if (idx >= 0) {
            // Target index found.
            destEntries = data.clone();
            destIndices = indices.clone();
            destEntries[idx] = value;
            destIndices[idx] = target[0];
        } else {
            // Target not found, insert new value and index.
            destEntries = makeEmptyDataArray(nnz + 1);
            destIndices = new int[nnz + 1];
            int insertionPoint = - (idx + 1);
            CooGetSet.cooInsertNewValue(value, target[0], data, indices, insertionPoint, destEntries, destIndices);
        }

        return makeLikeTensor(shape, destEntries, destIndices);
    }


    /**
     * Flattens tensor to single dimension while preserving order of data.
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
     * Flattens a tensor along the specified axis. Unlike {@link #flatten()}
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        return copy();
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
    public T reshape(Shape newShape) {
        ValidateParameters.ensureRank(newShape, 1);
        ValidateParameters.ensureTotalEntriesEqual(shape, newShape);
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
        Y[] destEntries = makeEmptyDataArray(this.data.length + b.data.length);
        int[] destIndices = new int[this.indices.length + b.indices.length];
        CooConcat.join(data, indices, size, b.data, b.indices, destEntries, destIndices);
        return makeLikeTensor(new Shape(shape.get(0) + b.shape.get(0)), destEntries, destIndices);
    }


    /**
     * <p>Computes the inner product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #dot(AbstractCooSemiringVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #dot(AbstractCooSemiringVector) 
     */
    @Override
    public Y inner(T b) {
        return dot(b); // For semirings, this will be the same.
    }


    /**
     * <p>Computes the dot product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #inner(AbstractCooSemiringVector)}. 
     * The inner product is equivalent to the dot product of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #inner(AbstractCooSemiringVector) 
     */
    @Override
    public Y dot(T b) {
        return CooSemiringVectorOps.dot(shape, data, indices, b.shape, b.data, b.indices);
    }


    /**
     * <p>Gets the length of a vector. Same as {@link #size()}.
     * <p>WARNING: This method will throw a {@link ArithmeticException} if the
     * total number of data in this vector is greater than the maximum integer. In this case, the true size of this vector can
     * still be found by calling {@code shape.totalEntries()} on this vector.
     *
     * @return The length, i.e. the number of data, in this vector.
     * @throws ArithmeticException If the total number of data in this vector is greater than the maximum integer.
     */
    @Override
    public int length() {
        return shape.totalEntriesIntValueExact();
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
    public V repeat(int n, int axis) {
        Y[] tiledEntries = makeEmptyDataArray(n*data.length);
        int[] tiledRows = new int[tiledEntries.length];
        int[] tiledCols = new int[tiledEntries.length];
        Shape tiledShape = CooConcat.repeat(data, indices, size, n, axis, tiledEntries, tiledRows, tiledCols);
        return makeLikeMatrix(tiledShape, tiledEntries, tiledRows, tiledCols);
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2&times;n} matrix.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n&times;2} matrix.
     * 
     *
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of
     *                                  data in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public V stack(T b, int axis) {
        ValidateParameters.ensureAllEqual(size, b.size);
        Y[] destEntries = makeEmptyDataArray(data.length + b.data.length);
        int[][] destIndices = new int[2][indices.length + b.indices.length]; // Row and column indices.

        CooConcat.stack(data, indices, b.data, b.indices, destEntries, destIndices[0], destIndices[1]);
        V mat = makeLikeMatrix(new Shape(2, size), destEntries, destIndices[0], destIndices[1]);

        return (axis == 0) ? mat : mat.T();
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of data.
     */
    @Override
    public W outer(T b) {
        Shape destShape = new Shape(size, b.size);
        Y[] dest = makeEmptyDataArray(size*b.size);
        CooSemiringVectorOps.outerProduct(data, indices, size, b.data, b.indices, dest);
        return makeLikeDenseMatrix(shape, dest);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public V toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[data.length];
            Shape matShape = new Shape(size, 1);

            return makeLikeMatrix(matShape, data.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[data.length];
            int[] colIndices = indices.clone();
            Shape matShape = new Shape(1, size);

            return makeLikeMatrix(matShape, data.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        SparseVectorData<Y> result = CooSemiringVectorOps.add(
                shape, data, indices, b.shape, b.data, b.indices);
        return makeLikeTensor(shape, result.data(), result.indices());
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
        SparseVectorData<Y> prod = CooSemiringVectorOps.elemMult(
                shape, data, indices,
                b.shape, b.data, b.indices);
        return makeLikeTensor(shape, prod.data(), prod.indices());
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
    public U tensorDot(T src2, int[] aAxes, int[] bAxes) {
        if(aAxes.length != 1 || bAxes.length != 1) {
            throw new LinearAlgebraException("Vector dot product requires exactly one dimension for each vector but got "
                    + aAxes.length + " and " + bAxes.length + ".");
        }
        if(aAxes[0] != 0 || bAxes[0] != 0) {
            throw new LinearAlgebraException("Both axes must be 0 for vector dot product but got "
                    + aAxes[0] + " and " + bAxes[0] + ".");
        }

        return makeLikeDenseTensor(shape, dot(src2));
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        throw new LinearAlgebraException("Tensor trace cannot be computed for a rank 1 tensor " +
                "(must be rank 2 or " + "greater).");
    }


    /**
     * Gets the zero element for the field of this vector.
     * @return The zero element for the field of this vector. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Semiring)} then {@code null} will be returned.
     */
    public Y getZeroElement() {
        return zeroElement;
    }


    /**
     * Sets the zero element for the field of this tensor.
     * @param zeroElement The zero element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the semiring.
     */
    public void setZeroElement(Y zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }


    /**
     * Converts this sparse COO matrix to an equivalent dense matrix.
     * @return A dense matrix equivalent to this sparse COO matrix.
     */
    public U toDense() {
        Y[] entries = makeEmptyDataArray(shape.totalEntriesIntValueExact());
        Arrays.fill(entries, zeroElement);

        for(int i = 0; i< nnz; i++)
            entries[indices[i]] = this.data[i];

        return makeLikeDenseTensor(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public abstract AbstractTensor<?, Y[], Y> toTensor();


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
     * @param newShape New shape for the tensor. Can be any rank but must be broadcastable to {@link #shape this.shape}.
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    public abstract AbstractTensor<?,  Y[], Y> toTensor(Shape newShape);


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public T normalize() {
        throw new UnsupportedOperationException("Normalization not supported for semiring vectors.");
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public Y mag() {
        return AggregateSemiring.sum(data);
    }


    /**
     * Gets the element of this vector at the specified index.
     *
     * @param idx Index of the element to get within this vector.
     *
     * @return The element of this vector at index {@code idx}.
     */
    @Override
    public Y get(int idx) {
        ValidateParameters.validateTensorIndex(shape, idx);
        Y value = CooGetSet.getCoo(data, indices, idx);
        return (value == null) ? getZeroElement() : value;
    }


    /**
     * Coalesces this sparse COO vector. An uncoalesced vector is a sparse vector with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by summing duplicated data. If another form of aggregation other
     * than summing is desired, use {@link #coalesce(BinaryOperator)}.
     * @return A new coalesced sparse COO vector which is equivalent to this COO vector.
     * @see #coalesce(BinaryOperator)
     */
    public T coalesce() {
        SparseVectorData<Y> vec = SparseUtils.coalesce(Semiring::add, shape, data, indices);
        return makeLikeTensor(vec.shape(), vec.data(), vec.indices());
    }


    /**
     * Coalesces this sparse COO vector. An uncoalesced vector is a sparse vector with multiple data for a single index. This
     * method will ensure that each index only has one non-zero value by aggregating duplicated data using {@code aggregator}.
     * @param aggregator Custom aggregation function to combine multiple.
     * @return A new coalesced sparse COO vector which is equivalent to this COO vector.
     * @see #coalesce()
     */
    public T coalesce(BinaryOperator<Y> aggregator) {
        SparseVectorData<Y> vec = SparseUtils.coalesce(aggregator, shape, data, indices);
        return makeLikeTensor(vec.shape(), vec.data(), vec.indices());
    }


    /**
     * Drops any explicit zeros in this sparse COO vector.
     * @return A copy of this COO vector with any explicitly stored zeros removed.
     */
    public T dropZeros() {
        SparseVectorData<Y> vec = SparseUtils.dropZeros(shape, data, indices);
        return makeLikeTensor(vec.shape(), vec.data(), vec.indices());
    }
}
