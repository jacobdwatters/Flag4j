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

package org.flag4j.arrays.backend_new.field;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.AbstractTensor;
import org.flag4j.arrays.backend_new.SparseVectorData;
import org.flag4j.arrays.backend_new.VectorMixin;
import org.flag4j.linalg.operations.common.field_ops.FieldOps;
import org.flag4j.linalg.operations.sparse.coo.CooConcat;
import org.flag4j.linalg.operations.sparse.coo.CooDataSorter;
import org.flag4j.linalg.operations.sparse.coo.CooGetSet;
import org.flag4j.linalg.operations.sparse.coo.ring_ops.CooRingVectorOps;
import org.flag4j.linalg.operations.sparse.coo.semiring_ops.CooSemiringVectorOps;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A sparse vector stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
 * elements of a {@link Field}.</p>
 *
 * <p>The {@link #entries non-zero entries} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse vectors allow for the efficient storage of and operations on large vectors that contain many zero values.</p>
 *
 * <p>COO vectors are optimized for large hyper-sparse vectors (i.e. vectors which contain almost all zeros relative to the size of the
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
 * @param <U> Type of equivalent dense vector.
 * @param <V> Type of matrix equivalent to {@code T}.
 * @param <W> Type of dense matrix equivalent to {@code U}.
 * @param <Y> Type of the field element in this vector.
 */
public abstract class AbstractCooFieldVector<
        T extends AbstractCooFieldVector<T, U, V, W, Y>,
        U extends AbstractDenseFieldVector<U, W, Y>,
        V extends AbstractCooFieldMatrix<V, W, T, Y>,
        W extends AbstractDenseFieldMatrix<W, U, Y>,
        Y extends Field<Y>>
        extends AbstractTensor<T, Field<Y>[], Y>
        implements FieldTensorMixin<T, U, Y>, VectorMixin<T, V, W, Y> {

    /**
     * The zero element for the ring that this tensor's elements belong to.
     */
    private Field<Y> zeroElement;
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
     * The sparsity of this matrix.
     */
    public final double sparsity;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected AbstractCooFieldVector(Shape shape, Field<Y>[] entries, int[] indices) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 1);
        ValidateParameters.ensureIndexInBounds(shape.get(0), indices);
        this.size = shape.totalEntriesIntValueExact();

        if(entries.length != indices.length) {
            throw new IllegalArgumentException("entries and indices arrays of a COO vector must have the same length but got " +
                    "lengths" + entries.length + " and " + indices.length + ".");
        }
        if(entries.length > size) {
            throw new IllegalArgumentException("The number of entries cannot be greater than the size of the vector but but got " +
                    "entries.length=" + entries.length + " and size=" + size + ".");
        }

        this.indices = indices;
        this.nnz = entries.length;
        sparsity = BigDecimal.valueOf(nnz).divide(new BigDecimal(shape.totalEntries())).doubleValue();

        // Attempt to set the zero element for the ring.
        this.zeroElement = (entries.length > 0) ? entries[0].getZero() : null;
    }


    /**
     * Constructs a sparse COO vector of the same type as this vector with the specified non-zero entries and indices.
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero entries of the vector to construct.
     * @param indices Non-zero row indices of the vector to construct.
     * @return A sparse COO vector of the same type as this vector with the specified non-zero entries and indices.
     */
    public abstract T makeLikeTensor(Shape shape, Field<Y>[] entries, int[] indices);


    /**
     * Constructs a dense vector of a similar type as this vector with the specified shape and entries.
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the vector to construct.
     * @return A dense vector of a similar type as this vector with the specified entries.
     */
    public abstract U makeLikeDenseTensor(Shape shape, Field<Y>... entries);


    /**
     * Constructs a dense matrix of a similar type as this vector with the specified shape and entries.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     * @return A dense matrix of a similar type as this vector with the specified entries.
     */
    public abstract W makeLikeDenseMatrix(Shape shape, Field<Y>... entries);


    /**
     * Constructs a COO vector with the specified shape, non-zero entries, and non-zero indices.
     * @param shape Shape of the vector.
     * @param entries Non-zero values of the vector.
     * @param indices Indices of the non-zero values in the vector.
     * @return A COO vector of the same type as this vector with the specified shape, non-zero entries, and non-zero indices.
     */
    public abstract T makeLikeTensor(Shape shape, List<Field<Y>> entries, List<Integer> indices);


    /**
     * Constructs a COO matrix with the specified shape, non-zero entries, and row and column indices.
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero entries of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     * @return A COO matrix of similar type as this vector with the specified shape, non-zero entries, and non-zero row/col indices.
     */
    public abstract V makeLikeMatrix(Shape shape, Field<Y>[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Gets the sparsity of this matrix as a decimal percentage.
     * That is, the percentage of entries in this matrix that are zero.
     * @return The sparsity of this matrix as a decimal percentage.
     * @see #density()
     */
    public double sparsity() {
        return sparsity;
    }


    /**
     * Gets the density of this matrix as a decimal percentage.
     * That is, the percentage of entries in this matrix that are non-zero.
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
        CooDataSorter.wrap(entries, indices).sparseSort().unwrap(entries, indices);
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
        ValidateParameters.validateTensorIndex(shape, target);
        Y value = (Y) CooGetSet.getCoo(entries, indices, target[0]);
        return (value == null) ? (Y) getZeroElement() : value;
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
        return makeLikeTensor(shape, entries);
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

        Field<Y>[] destEntries;
        int[] destIndices;

        if (idx >= 0) {
            // Target index found.
            destEntries = entries.clone();
            destIndices = indices.clone();
            destEntries[idx] = value;
            destIndices[idx] = target[0];
        } else {
            // Target not found, insert new value and index.
            destEntries = new Field[nnz + 1];
            destIndices = new int[nnz + 1];
            int insertionPoint = - (idx + 1);
            CooGetSet.cooInsertNewValue(value, target[0], entries, indices, insertionPoint, destEntries, destIndices);
        }

        return makeLikeTensor(shape, (Y[]) destEntries, destIndices);
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
     * Flattens a tensor along the specified axis. Unlike {@link #flatten()}
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
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
        Field<Y>[] destEntries = new Field[this.entries.length + b.entries.length];
        int[] destIndices = new int[this.indices.length + b.indices.length];
        CooConcat.join(entries, indices, size, b.entries, b.indices, destEntries, destIndices);
        return makeLikeTensor(new Shape(shape.get(0) + b.shape.get(0)), destEntries, destIndices);
    }


    /**
     * <p>Computes the inner product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #dot(AbstractCooFieldVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #dot(AbstractCooFieldVector)
     */
    @Override
    public Y inner(T b) {
        // TODO: Implementation.
        return dot(b); // For rings, this will be the same.
    }


    /**
     * <p>Computes the dot product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #inner(AbstractCooFieldVector)}.
     * The inner product is equivalent to the dot product of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #inner(AbstractCooFieldVector)
     */
    @Override
    public Y dot(T b) {
        return (Y) CooSemiringVectorOps.dot(shape, entries, indices, b.shape, b.entries, b.indices);
    }


    /**
     * <p>Gets the length of a vector. Same as {@link #size()}.</p>
     * <p>WARNING: This method will throw a {@link ArithmeticException} if the
     * total number of entries in this vector is greater than the maximum integer. In this case, the true size of this vector can
     * still be found by calling {@code shape.totalEntries()} on this vector.</p>
     *
     * @return The length, i.e. the number of entries, in this vector.
     * @throws ArithmeticException If the total number of entries in this vector is greater than the maximum integer.
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
        Field<Y>[] tiledEntries = new Field[n*entries.length];
        int[] tiledRows = new int[tiledEntries.length];
        int[] tiledCols = new int[tiledEntries.length];
        Shape tiledShape = CooConcat.repeat(entries, indices, size, n, axis, tiledEntries, tiledRows, tiledCols);
        return makeLikeMatrix(tiledShape, entries, tiledRows, tiledCols);
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
     * @param b Vector to stack with this vector.
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
    public V stack(T b, int axis) {
        ValidateParameters.ensureEquals(size, b.size);
        Field<Y>[] destEntries = new Field[entries.length + b.entries.length];
        int[][] destIndices = new int[2][indices.length + indices.length]; // Row and column indices.

        CooConcat.stack(entries, indices, b.entries, b.indices, destEntries, destIndices[0], destIndices[1]);
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
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public W outer(T b) {
        Shape destShape = new Shape(size, b.size);
        Field<Y>[] dest = new Field[size*b.size];
        CooSemiringVectorOps.outerProduct(entries, indices, size, b.entries, b.indices, dest);
        return makeLikeDenseMatrix(shape, dest);
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
    public V toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];
            Shape matShape = new Shape(size, 1);

            return makeLikeMatrix(matShape, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();
            Shape matShape = new Shape(1, size);

            return makeLikeMatrix(matShape, entries.clone(), rowIndices, colIndices);
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
        SparseVectorData<Semiring<Y>> result = CooSemiringVectorOps.add(
                shape, entries, indices, b.shape, b.entries, b.indices);
        return makeLikeTensor(shape,
                (Y[]) result.entries().toArray(new Field[result.entries().size()]),
                ArrayUtils.fromIntegerList(result.indices()));
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
        SparseVectorData<Semiring<Y>> prod = CooSemiringVectorOps.elemMult(entries, indices, b.entries, b.indices);
        return makeLikeTensor(shape,
                (Y[]) prod.entries().toArray(new Field[prod.entries().size()]),
                ArrayUtils.fromIntegerList(prod.indices()));
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
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays of this tensor specified by
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
     * Gets the zero element for the field of this vector.
     * @return The zero element for the field of this vector. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Field)} then {@code null} will be returned.
     */
    public Y getZeroElement() {
        return (Y) zeroElement;
    }


    /**
     * Sets the zero element for the field of this tensor.
     * @param zeroElement The zero element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the ring.
     */
    public void setZeroElement(Field<Y> zeroElement) {
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
        Field<Y>[] entries = new Field[shape.totalEntriesIntValueExact()];

        for(int i = 0; i< nnz; i++)
            entries[indices[i]] = this.entries[i];

        return makeLikeDenseTensor(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     * @return A tensor which is equivalent to this matrix.
     */
    public abstract AbstractTensor<?, Field<Y>[], Y> toTensor();


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
     * @param newShape New shape for the tensor. Can be any rank but must be broadcastable to {@link #shape this.shape}.
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    public abstract AbstractTensor<?,  Field<Y>[], Y> toTensor(Shape newShape);


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T sub(T b) {
        SparseVectorData<Ring<Y>> result = CooRingVectorOps.sub(
                shape, entries, indices, b.shape, b.entries, b.indices);
        return makeLikeTensor(shape,
                (Y[]) result.entries().toArray(new Field[result.entries().size()]),
                result.indicesToArray());
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
    public T H(int axis1, int axis2) {
        return T(axis1, axis2);
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
        return T(axes);
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     * <p><b>WARNING</b>: This method is not supported for sparse tensors. If called on a sparse tensor,
     * an {@link UnsupportedOperationException} will be thrown. Element-wise division is undefined for sparse matrices as it
     * would almost certainly result in a division by zero.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     * @throws UnsupportedOperationException if this method is ever invoked on a sparse tensor.
     */
    @Override
    public T div(T b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse vectors.");
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        Field<Y>[] dest = new Field[entries.length];
        FieldOps.sqrt(entries, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values. Otherwise, returns {@code false}.
     *
     * @see #isInfinite()
     * @see #isNaN()
     */
    @Override
    public boolean isFinite() {
        return FieldOps.isFinite(entries);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value. Otherwise, returns {@code false}.
     *
     * @see #isFinite()
     * @see #isNaN()
     */
    @Override
    public boolean isInfinite() {
        return FieldOps.isInfinite(entries);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value. Otherwise, returns {@code false}.
     *
     * @see #isFinite()
     * @see #isInfinite()
     */
    @Override
    public boolean isNaN() {
        return FieldOps.isNaN(entries);
    }
}
