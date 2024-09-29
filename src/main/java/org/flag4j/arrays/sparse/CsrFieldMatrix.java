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
import org.flag4j.arrays.backend.CsrFieldMatrixBase;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.operations.sparse.SparseUtils;


/**
 * <p>A sparse matrix stored in compressed sparse row (CSR) format. The {@link #entries} of this CSR matrix are
 * elements of a {@link Field}.</p>
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a CSR matrix are mutable but the {@link #shape}
 * and {@link #nnz total number of non-zero entries} is fixed.</p>
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero values.</p>
 *
 * <p>A sparse CSR matrix is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowPointers row pointers} of the non-zero values in the CSR matrix. Has size {@link #numRows numRows + 1}</li>
 *     <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all values in row
 *     {@code i}.</p>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the CSR matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any operations implemented in this
 * class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted explicitly, call {@link #sortIndices()}.</p>
 */
public class CsrFieldMatrix<T extends Field<T>> extends CsrFieldMatrixBase<CsrFieldMatrix<T>, FieldMatrix<T>,
        CooFieldVector<T>, FieldVector<T>, T> {


    /**
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    public CsrFieldMatrix(Shape shape, Field<T>[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries, rowPointers, colIndices);
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
    public CsrFieldMatrix<T> set(T value, int... indices) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Constructs a CSR matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and the same non-zero
     * indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and the same non-zero
     * indices.
     */
    @Override
    public CsrFieldMatrix<T> makeLikeTensor(Shape shape, Field<T>[] entries) {
        return new CsrFieldMatrix<T>(shape, entries, rowPointers.clone(), colIndices.clone());
    }


    /**
     * Constructs a CSR matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and non-zero
     * indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     * @param rowPointers Row pointers for the CSR matrix.
     * @param colIndices Column indices of the CSR matrix.
     *
     * @return A matrix of the same type as this matrix with the given the {@code shape}, {@code entries} and non-zero
     * indices.
     */
    @Override
    public CsrFieldMatrix<T> makeLikeTensor(Shape shape, Field<T>[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrFieldMatrix<>(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a dense matrix of similar type as this matrix with the given the {@code shape} and {@code entries}.
     *
     * @param shape Shape of the dense matrix to construct.
     * @param entries Entries of the dense matrix to construct.
     *
     * @return A dense matrix of similar type as this sparse CSR matrix with the given the {@code shape} and {@code entries}.
     */
    @Override
    public FieldMatrix<T> makeLikeDenseTensor(Shape shape, Field<T>[] entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooFieldMatrix<T> toCoo() {
        int[] cooRowIdx = new int[entries.length];

        for(int i=0; i<numRows; i++) {
            int stop = rowPointers[i+1];

            for(int j=rowPointers[i]; j<stop; j++)
                cooRowIdx[j] = i;
        }

        return new CooFieldMatrix<T>(shape, entries.clone(), cooRowIdx, colIndices.clone());
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CsrFieldMatrix<T> b = (CsrFieldMatrix<T>) object;

        return SparseUtils.CSREquals(this, b);
    }


    @Override
    public int hashCode() {
        if(nnz == 0) return 0;

        int result = 17;

        // Hash calculation ignores explicit zeros in the matrix. This upholds the contract with the equals(Object) method.
        for(int row = 0; row<numRows; row++) {
            for(int idx = rowPointers[row], rowStop = rowPointers[row + 1]; idx < rowStop; idx++) {
                if(!entries[idx].isZero()) {
                    result = 31*result + entries[idx].hashCode();
                    result = 31*result + Integer.hashCode(colIndices[idx]);
                    result = 31*result + Integer.hashCode(row);
                }
            }
        }

        return result;
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


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooFieldVector<T> toVector() {
        // TODO: Implement this method
        return null;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     *
     * @return The specified row of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     *                                        the number of rows in this matrix.
     */
    @Override
    public CooFieldVector<T> getRow(int rowIdx) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Gets a specified row of this matrix between {@code colStart} (inclusive) and {@code colEnd} (exclusive).
     *
     * @param rowIdx Index of the row of this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The row at index {@code rowIdx} of this matrix between the {@code colStart} and {@code colEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code colEnd} is less than {@code colStart}.
     */
    @Override
    public CooFieldVector<T> getRow(int rowIdx, int colStart, int colEnd) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     *
     * @return The specified column of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     *                                        the number of columns in this matrix.
     */
    @Override
    public CooFieldVector<T> getCol(int colIdx) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param colIdx Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     *
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     *
     * @throws @throws                  IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the
     *                                  shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public CooFieldVector<T> getCol(int colIdx, int rowStart, int rowEnd) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public CooFieldVector<T> getDiag() {
        // TODO: Implement this method
        return null;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrFieldMatrix<T> setCol(CooFieldVector<T> values, int colIndex) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrFieldMatrix<T> setRow(CooFieldVector<T> values, int rowIndex) {
        // TODO: Implement this method
        return null;
    }
}
