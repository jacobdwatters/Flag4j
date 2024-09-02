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
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.util.ParameterChecks;

import java.util.List;

/**
 * <p>A complex sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * {@link Complex128}'s.</p>
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
 */
public class CooCMatrix extends CooFieldMatrixBase<CooCMatrix, CMatrix, CooCVector, Complex128> {

    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries, rowIndices, colIndices);
        ParameterChecks.ensureRank(shape, 2);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, List<Complex128> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape, entries, rowIndices, colIndices);
        ParameterChecks.ensureRank(shape, 2);
        if(super.entries.length == 0 || super.entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param shape The shape of the matrix.
     */
    public CooCMatrix(Shape shape) {
        super(shape, new Complex128[0], new int[0], new int[0]);
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int rows, int cols, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int rows, int cols, List<Complex128> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
        if(super.entries.length == 0 || super.entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    public CooCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[0], new int[0], new int[0]);
        ParameterChecks.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
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
    public CooCMatrix makeLikeTensor(Shape shape, Complex128[] entries) {
        return new CooCMatrix(shape, entries, rowIndices.clone(), colIndices.clone());
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
    public CooCMatrix makeLikeTensor(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
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
    public CooCMatrix makeLikeTensor(Shape shape, List<Complex128> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
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
    public CMatrix makeDenseTensor(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
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
    public CooCVector makeLikeVector(int size, Complex128[] entries, int[] indices) {
        return new CooCVector(size, entries, indices);
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
    public CooCVector makeLikeVector(int size, List<Complex128> entries, List<Integer> indices) {
        return new CooCVector(size, entries, indices);
    }


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    @Override
    public CsrCMatrix toCsr() {
        int[] rowPointers = new int[numRows + 1];

        // Count number of entries per row.
        for(int i=0; i<nnz; i++)
            rowPointers[rowIndices[i] + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<numRows; i++)
            rowPointers[i+1] += rowPointers[i];

        return new CsrCMatrix(shape, entries.clone(), rowPointers, colIndices.clone());
    }
}
