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

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldVectorBase;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.operations.sparse.coo.field_ops.SparseFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

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
 * @param <T> Type of the field element in this vector.
 */
public class CooFieldVector<T extends Field<T>> extends CooFieldVectorBase<CooFieldVector<T>, CooFieldMatrix<T>,
        FieldVector<T>, FieldMatrix<T>, T> {


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero entries of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooFieldVector(int size, Field<T>[] entries, int[] indices) {
        super(size, entries, indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero entries of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooFieldVector(int size, List<Field<T>> entries, List<Integer> indices) {
        super(size, (T[]) entries.toArray(Field[]::new), ArrayUtils.fromIntegerList(indices));
    }


    /**
     * Creates a zero vector of the specified {@code size}.
     */
    public CooFieldVector(int size) {
        super(size, new Field[0], new int[0]);
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     *
     * @return A sparse COO vector of the same type as this vector with the specified {@code size}, non-zero entries,
     * and non-zero indices.
     */
    @Override
    public CooFieldVector<T> makeLikeTensor(int size, Field<T>[] entries, int[] indices) {
        return new CooFieldVector<T>(size, entries, indices);
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and the same
     * non-zero indices as this vector.
     *
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     *
     * @return A sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and the same
     * non-zero indices as this vector.
     */
    @Override
    public CooFieldVector<T> makeLikeTensor(int size, Field<T>[] entries) {
        return new CooFieldVector<T>(size, entries, indices.clone());
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     *
     * @return A sparse COO vector of the same type as this vector with the specified {@code size}, non-zero entries,
     * and non-zero indices.
     */
    @Override
    public CooFieldVector<T> makeLikeTensor(int size, List<Field<T>> entries, List<Integer> indices) {
        return new CooFieldVector<T>(size, entries, indices);
    }


    /**
     * Constructs a dense vector which is of a similar type to this sparse COO vector containing the specified {@code entries}.
     *
     * @param entries The entries of the dense vector.
     *
     * @return A dense vector which is of a similar type to this sparse COO vector containing the specified {@code entries}.
     */
    @Override
    public FieldVector<T> makeLikeDenseTensor(Field<T>... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Constructs a sparse matrix which is of a similar type to this sparse COO vector with the specified {@code shape}, non-zero
     * entries, non-zero row indices, and non-zero column indices.
     *
     * @param shape Shape of the matrix.
     * @param entries The non-zero indices of the matrix.
     * @param rowIndices The row indices of the non-zero entries.
     * @param colIndices The column indices of the non-zero entries.
     *
     * @return A dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     */
    @Override
    public CooFieldMatrix<T> makeLikeMatrix(Shape shape, Field<T>[] entries, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     *
     * @param shape Shape of the dense matrix.
     * @param entries The entries of the dense matrix.
     *
     * @return A dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     */
    @Override
    public FieldMatrix<T> makeLikeDenseMatrix(Shape shape, Field<T>... entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Constructs a sparse COO vector of the specified size filled with zeros.
     *
     * @param size The size of the vector to construct.
     *
     * @return A sparse COO vector of the specified size filled with zeros.
     */
    @Override
    public CooFieldVector<T> makeZeroVector(int size) {
        return new CooFieldVector<T>(size);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooFieldVector<T> set(T value, int... indices) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape, non-zero entries and the same non-zero indices
     * as this vector.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooFieldVector<T> makeLikeTensor(Shape shape, Field<T>[] entries) {
        ValidateParameters.ensureRank(shape, 1);
        return new CooFieldVector<T>(shape.totalEntriesIntValueExact(), entries, indices.clone());
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooFieldVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooFieldVector<T> src2 = (CooFieldVector<T>) object;

        return SparseFieldEquals.cooVectorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (!entries[i].isZero()) {
                result = 31*result + entries[i].hashCode();
                result = 31*result + Integer.hashCode(indices[i]);
            }
        }

        return result;
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
    public CooFieldMatrix<T> toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new CooFieldMatrix<T>(this.size, 1, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new CooFieldMatrix<T>(1, this.size, entries.clone(), rowIndices, colIndices);
        }
    }
}
