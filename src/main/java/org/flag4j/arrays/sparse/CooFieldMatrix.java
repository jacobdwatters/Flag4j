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
import org.flag4j.arrays.backend.CooFieldMatrixBase;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldMatrixGetSet;
import org.flag4j.util.ValidateParameters;

import java.util.List;


/**
 * <p>A sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * elements of a {@link Field}.</p>
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero values.</p>
 *
 * <p>COO matrices are optimized for hyper-sparse matrices (i.e. matrices which contain almost all zeros relative to the size of the
 * matrix).</p>
 *
 * <p>A sparse COO matrix is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified but any operations implemented in this
 * class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.</p>
 *
 * @param <T> Type of the {@link Field field} element in this matrix.
 */
public class CooFieldMatrix<T extends Field<T>> extends CooFieldMatrixBase<CooFieldMatrix<T>,
        FieldMatrix<T>, CooFieldVector<T>, FieldVector<T>, T> {

    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooFieldMatrix(Shape shape, Field<T>[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooFieldMatrix(Shape shape, List<Field<T>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape, entries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows Rows in the coo matrix.
     * @param cols Columns in the coo matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooFieldMatrix(int rows, int cols, Field<T>[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows Rows in the coo matrix.
     * @param cols Columns in the coo matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooFieldMatrix(int rows, int cols, List<Field<T>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
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
    public CooFieldMatrix<T> set(T value, int... indices) {
        ValidateParameters.ensureValidIndex(shape, indices);
        return (CooFieldMatrix<T>) CooFieldMatrixGetSet.matrixSet(this, indices[0], indices[1], value);
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
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, Field<T>[] entries) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Constructs a COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Row indices of the non-zero values in the matrix.
     * @param colIndices Column indices of the non-zero values in the matrix.
     *
     * @return A COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, Field<T>[] entries, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Row indices of the non-zero values in the matrix.
     * @param colIndices Column indices of the non-zero values in the matrix.
     *
     * @return A COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CooFieldMatrix<T> makeLikeTensor(Shape shape, List<Field<T>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a dense field matrix which with the specified {@code shape} and {@code entries}.
     *
     * @param shape Shape of the matrix.
     * @param entries Entries of the dense matrix/.
     *
     * @return A dense field matrix with the specified {@code shape} and {@code entries}.
     */
    @Override
    public FieldMatrix<T> makeDenseTensor(Shape shape, Field<T>[] entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Constructs a vector of similar type to this matrix.
     *
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero values of the vector.
     *
     * @return A vector of similar type to this matrix with the specified size, non-zero entries, and indices.
     */
    @Override
    public CooFieldVector<T> makeLikeVector(int size, Field<T>[] entries, int[] indices) {
        return new CooFieldVector<T>(size, entries, indices);
    }


    /**
     * Constructs a vector of similar type to this matrix.
     *
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero values of the vector.
     *
     * @return A vector of similar type to this matrix with the specified size, non-zero entries, and indices.
     */
    @Override
    public CooFieldVector<T> makeLikeVector(int size, List<Field<T>> entries, List<Integer> indices) {
        return new CooFieldVector<T>(size, entries, indices);
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooFieldMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooFieldMatrix<T> src2 = (CooFieldMatrix<T>) object;

        return CooFieldEquals.cooMatrixEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (!entries[i].isZero()) {
                result = 31*result + entries[i].hashCode();
                result = 31*result + Integer.hashCode(rowIndices[i]);
                result = 31*result + Integer.hashCode(colIndices[i]);
            }
        }

        return result;
    }


    /**
     * <p>Converts this sparse COO matrix to an equivalent compressed sparse row (CSR) matrix.</p>
     * <p>It is often easier and more efficient to construct a matrix in COO format first then convert to a CSR matrix for efficient
     * computations.</p>
     *
     * @return A CSR matrix equivalent to this COO matrix.
     */
    public CsrFieldMatrix<T> toCsr() {
        int[] rowPointers = new int[numRows + 1];

        // Count number of entries per row.
        for(int i=0; i<nnz; i++)
            rowPointers[rowIndices[i] + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<numRows; i++)
            rowPointers[i+1] += rowPointers[i];

        return new CsrFieldMatrix<T>(shape, entries.clone(), rowPointers, colIndices.clone());
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the
     *                                  number of entries in the vector {@code b}.
     */
    @Override
    public FieldVector<T> mult(CooFieldVector<T> b) {
        // TODO: Implement this method
        return null;
    }
}
