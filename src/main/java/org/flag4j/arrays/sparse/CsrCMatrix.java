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
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.field.AbstractCsrFieldMatrix;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.sparse.SparseUtils;
import org.flag4j.linalg.operations.sparse.csr.CsrConversions;
import org.flag4j.linalg.operations.sparse.csr.semiring_ops.SemiringCsrMatMult;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;
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
public class CsrCMatrix extends AbstractCsrFieldMatrix<CsrCMatrix, CMatrix, CooCVector, Complex128> {

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
    public CsrCMatrix(Shape shape, Field<Complex128>[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries, rowPointers, colIndices);
        ValidateParameters.ensureRank(shape, 2);
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
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param shape Shape of the zero matrix.
     */
    public CsrCMatrix(Shape shape) {
        super(shape, new Complex128[0], new int[0], new int[0]);
        ValidateParameters.ensureRank(shape, 2);
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
    public CsrCMatrix(int rows, int cols, Field<Complex128>[] entries, int[] rowPointers, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowPointers, colIndices);
        ValidateParameters.ensureRank(shape, 2);
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
    public CsrCMatrix(int rows, int cols, List<Field<Complex128>> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        super(new Shape(rows, cols), entries.toArray(new Complex128[0]),
                ArrayUtils.fromIntegerList(rowPointers),
                ArrayUtils.fromIntegerList(colIndices));
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    public CsrCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[0], new int[0], new int[0]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a sparse CSR tensor of the same type as this tensor with the specified non-zero entries and indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A sparse CSR tensor of the same type as this tensor with the specified non-zero entries and indices.
     */
    @Override
    public CsrCMatrix makeLikeTensor(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a CSR matrix with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A CSR matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CsrCMatrix makeLikeTensor(Shape shape, List<Complex128> entries, List<Integer> rowPointers, List<Integer> colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a dense matrix which is of a similar type to this sparse CSR matrix.
     *
     * @param shape Shape of the dense matrix.
     * @param entries Entries of the dense matrix.
     *
     * @return A dense matrix which is of a similar type to this sparse CSR matrix with the specified {@code shape}
     * and {@code entries}.
     */
    @Override
    public CMatrix makeLikeDenseTensor(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * <p>Constructs a sparse COO matrix of a similar type to this sparse CSR matrix.
     * <p>Note: this method constructs a new COO matrix with the specified entries and indices. It does <i>not</i> convert this matrix
     * to a CSR matrix. To convert this matrix to a sparse COO matrix use {@link #toCoo()}.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the sparse COO matrix.
     * @param colIndices Non-zero column indices of the Sparse COO matrix.
     *
     * @return A sparse COO matrix of a similar type to this sparse CSR matrix.
     */
    @Override
    public CooCMatrix makeLikeCooMatrix(Shape shape, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooCMatrix toCoo() {
        Complex128[] cooEntries = new Complex128[nnz];
        int[] cooRowIndices = new int[nnz];
        int[] cooColIndices = new int[nnz];
        CsrConversions.toCoo(shape, entries, rowPointers, colIndices, cooEntries, cooRowIndices, cooColIndices);
        return new CooCMatrix(shape, cooEntries, cooRowIndices, cooColIndices);
    }


    /**
     * Converts this CSR matrix to an equivalent sparse COO tensor.
     *
     * @return An sparse COO tensor equivalent to this CSR matrix.
     */
    @Override
    public CooCTensor toTensor() {
        return toCoo().toTensor();
    }


    /**
     * Converts this CSR matrix to an equivalent COO tensor with the specified shape.
     *
     * @param shape@return A COO tensor equivalent to this CSR matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooCTensor toTensor(Shape shape) {
        return toTensor(shape).reshape(shape);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code entries}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code entries}.
     */
    @Override
    public CsrCMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CsrCMatrix(shape, entries, rowPointers.clone(), colIndices.clone());
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
    public CTensor tensorDot(CsrCMatrix src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of {@code b}.
     */
    @Override
    public CVector mult(CooCVector b) {
        Complex128[] dest = new Complex128[b.size];
        SemiringCsrMatMult.standardVector(shape, entries, rowPointers, colIndices,
                b.size, b.entries, b.indices,
                dest, getZeroElement());
        return new CVector(dest);
    }


    /**
     * Gets a range of a row of this matrix.
     *
     * @param rowIdx The index of the row to get.
     * @param start The staring column of the row range to get (inclusive).
     * @param stop The ending column of the row range to get (exclusive).
     *
     * @return A vector containing the elements of the specified row over the range [start, stop).
     *
     * @throws IllegalArgumentException If {@code rowIdx < 0 || rowIdx >= this.numRows()} or {@code start < 0 || start >= numCols} or
     *                                  {@code stop < start || stop > numCols}.
     */
    @Override
    public CooCVector getRow(int rowIdx, int start, int stop) {
        return toCoo().getRow(rowIdx, start, stop);
    }


    /**
     * Gets a range of a column of this matrix.
     *
     * @param colIdx The index of the column to get.
     * @param start The staring row of the column range to get (inclusive).
     * @param stop The ending row of the column range to get (exclusive).
     *
     * @return A vector containing the elements of the specified column over the range [start, stop).
     *
     * @throws IllegalArgumentException If {@code colIdx < 0 || colIdx >= this.numCols()} or {@code start < 0 || start >= numRows} or
     *                                  {@code stop < start || stop > numRows}.
     */
    @Override
    public CooCVector getCol(int colIdx, int start, int stop) {
        return toCoo().getCol(colIdx, start, stop);
    }


    /**
     * Gets the elements of this matrix along the specified diagonal.
     *
     * @param diagOffset The diagonal to get within this matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     *
     * @return The elements of the specified diagonal as a vector.
     */
    @Override
    public CooCVector getDiag(int diagOffset) {
        return toCoo().getDiag(diagOffset);
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

        CsrCMatrix b = (CsrCMatrix) object;

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
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("Non-zero entries: [");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        if(entries.length > 0) {
            // Get entries up until the stopping point.
            for(int i=0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound((Complex128) entries[i], PrintOptions.getPrecision());
                width = PrintOptions.getPadding() + value.length();
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = PrintOptions.getPadding() + 3;
                value = "...";
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound((Complex128) entries[size-1], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Pointers: ").append(Arrays.toString(rowPointers)).append("\n");
        result.append("Col Indices: ").append(Arrays.toString(colIndices));

        return result.toString();
    }
}
