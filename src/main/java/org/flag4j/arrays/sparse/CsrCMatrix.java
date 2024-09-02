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

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldMatrixBase;
import org.flag4j.arrays.backend.CsrFieldMatrixBase;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.List;

/**
 * <p>A complex sparse matrix stored in compressed sparse row (CSR) format. The {@link #entries} of this CSR matrix are
 * {@link Complex128}'s.</p>
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
public class CsrCMatrix extends CsrFieldMatrixBase<CsrCMatrix, CMatrix, CooCVector, Complex128> {

    /**
     * Creates a complex sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column
     * indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    public CsrCMatrix(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries, rowPointers, colIndices);
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column
     * indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    public CsrCMatrix(Shape shape, List<Complex128> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        super(shape, entries.toArray(new Complex128[0]),
                ArrayUtils.fromIntegerList(rowPointers),
                ArrayUtils.fromIntegerList(colIndices));
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param shape Shape of the zero matrix.
     */
    public CsrCMatrix(Shape shape) {
        super(shape, new Complex128[0], new int[0], new int[0]);
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex sparse CSR matrix with the specified shape, non-zero entries, row pointers, and non-zero column
     * indices.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    public CsrCMatrix(int rows, int cols, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowPointers, colIndices);
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column
     * indices.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    public CsrCMatrix(int rows, int cols, List<Complex128> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        super(new Shape(rows, cols), entries.toArray(new Complex128[0]),
                ArrayUtils.fromIntegerList(rowPointers),
                ArrayUtils.fromIntegerList(colIndices));
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    public CsrCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[0], new int[0], new int[0]);
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
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
    public CsrCMatrix makeLikeTensor(Shape shape, Complex128[] entries) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
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
    public CsrCMatrix makeLikeTensor(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
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
    public CMatrix makeLikeDenseTensor(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooFieldMatrixBase toCoo() {
        int[] cooRowIdx = new int[entries.length];

        for(int i=0; i<numRows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++)
                cooRowIdx[j] = i;
        }

        return new CooCMatrix(shape, entries.clone(), cooRowIdx, colIndices.clone());
    }
}
