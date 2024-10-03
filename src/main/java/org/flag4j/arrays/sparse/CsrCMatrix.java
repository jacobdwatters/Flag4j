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
import org.flag4j.arrays.backend.CsrFieldMatrixBase;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.io.PrintOptions;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseMatrixMultiplication;
import org.flag4j.operations.sparse.SparseUtils;
import org.flag4j.operations.sparse.csr.complex.ComplexCsrMatrixMultiplication;
import org.flag4j.operations.sparse.csr.complex.ComplexCsrOperations;
import org.flag4j.operations.sparse.csr.real_complex.RealComplexCsrMatrixMultiplication;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
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
public class CsrCMatrix extends CsrFieldMatrixBase<CsrCMatrix, CMatrix, CooCVector, CVector, Complex128> {

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
    public CsrCMatrix(Shape shape, List<Field<Complex128>> entries, List<Integer> rowPointers, List<Integer> colIndices) {
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
    public CsrCMatrix set(Complex128 value, int... indices) {
        // Ensure indices are in bounds.
        ValidateParameters.ensureValidIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];

        Field<Complex128>[] newEntries;
        int[] newRowPointers = rowPointers.clone();
        int[] newColIndices;
        boolean found = false; // Flag indicating an element already exists in this matrix at the specified row and col.
        int loc = -1;

        if(rowPointers[row] < rowPointers[row+1]) {
            int start = rowPointers[row];
            int stop = rowPointers[row+1];

            loc = Arrays.binarySearch(colIndices, start, stop, col);
            found = loc >= 0;
        }

        if(found) {
            newEntries = Arrays.copyOf(entries, entries.length);
            newEntries[loc] = value;
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch
            newEntries = new Complex128[entries.length + 1];
            newColIndices = new int[entries.length + 1];

            // Copy old entries and insert new one.
            System.arraycopy(entries, 0, newEntries, 0, loc);
            newEntries[loc] = value;
            System.arraycopy(entries, loc, newEntries, loc+1, entries.length-loc);

            // Copy old column indices and insert new one.
            System.arraycopy(colIndices, 0, newColIndices, 0, loc);
            newColIndices[loc] = col;
            System.arraycopy(colIndices, loc, newColIndices, loc+1, entries.length-loc);

            // Increment row pointers.
            for(int i=row+1; i<rowPointers.length; i++) {
                newRowPointers[i]++;
            }
        }

        return new CsrCMatrix(shape, newEntries, newRowPointers, newColIndices);
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
    public CsrCMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
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
    public CsrCMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries, int[] rowPointers, int[] colIndices) {
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
    public CMatrix makeLikeDenseTensor(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public CooCMatrix toCoo() {
        int[] cooRowIdx = new int[entries.length];

        for(int i=0; i<numRows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++)
                cooRowIdx[j] = i;
        }

        return new CooCMatrix(shape, entries.clone(), cooRowIdx, colIndices.clone());
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
    public CVector mult(CooCVector b) {
        return ComplexCsrMatrixMultiplication.standardVector(this, b);
    }





    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooCVector toVector() {
        int type = vectorType();
        int[] indices = new int[entries.length];

        if(type == -1) {
            // Not a vector.
            for(int i=0; i<numRows; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i+1];
                int rowOffset = i*numCols;

                for(int j=start; j<stop; j++) {
                    indices[j] = rowOffset + colIndices[j];
                }
            }

        } else if(type <= 1) {
            // Row vector.
            System.arraycopy(colIndices, 0, indices, 0, colIndices.length);
        } else {
            // Column vector.
            for(int i=0; i<numRows; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i+1];

                for(int j=start; j<stop; j++) {
                    indices[j] = i;
                }
            }
        }

        return new CooCVector(shape.totalEntries().intValueExact(), entries.clone(), indices);
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
    public CooCVector getRow(int rowIdx) {
        ValidateParameters.ensureIndexInBounds(numRows, rowIdx);
        int start = rowPointers[rowIdx];

        Field<Complex128>[] destEntries = new Complex128[rowPointers[rowIdx + 1]-start];
        int[] destIndices = new int[destEntries.length];

        System.arraycopy(entries, start, destEntries, 0, destEntries.length);
        System.arraycopy(colIndices, start, destIndices, 0, destEntries.length);

        return new CooCVector(this.numCols, destEntries, destIndices);
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
    public CooCVector getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.ensureIndexInBounds(numRows, rowIdx);
        ValidateParameters.ensureIndexInBounds(numCols, colStart, colEnd-1);
        int start = rowPointers[rowIdx];
        int end = rowPointers[rowIdx+1];

        List<Field<Complex128>> row = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int j=start; j<end; j++) {
            int col = colIndices[j];

            if(col >= colStart && col < colEnd) {
                row.add(entries[j]);
                indices.add(col-colStart);
            }
        }

        return new CooCVector(colEnd-colStart, row, indices);
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
    public CooCVector getCol(int colIdx) {
        return getCol(colIdx, 0, numRows);
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
    public CooCVector getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.ensureIndexInBounds(numCols, colIdx);
        ValidateParameters.ensureIndexInBounds(numRows, rowStart, rowEnd-1);

        List<Field<Complex128>> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=rowStart; i<rowEnd; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                if(colIndices[j]==colIdx) {
                    destEntries.add(entries[j]);
                    destIndices.add(i);
                    break; // Should only be a single entry with this row and column index.
                }
            }
        }

        return new CooCVector(numRows, destEntries, destIndices);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public CooCVector getDiag() {
        List<Field<Complex128>> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<numRows; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            int loc = Arrays.binarySearch(colIndices, start, stop, i); // Search for matching column index within row.

            if(loc >= 0) {
                destEntries.add(entries[loc]);
                destIndices.add(i);
            }
        }

        return new CooCVector(Math.min(numRows, numCols), destEntries, destIndices);
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
    public CsrCMatrix add(CsrCMatrix b) {
        return ComplexCsrOperations.applyBinOpp(this, b, Complex128::add, null);
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
    public CsrCMatrix sub(CsrCMatrix b) {
        return ComplexCsrOperations.applyBinOpp(this, b, Complex128::add, Complex128::addInv);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A copy of this matrix with the column at {@code colIndex} set to {@code values}.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrCMatrix setCol(CooCVector values, int colIndex) {
        // Convert to COO first for more efficient modification.
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A copy of this matrix with the row at {@code rowIndex} set to {@code values}.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrCMatrix setRow(CooCVector values, int rowIndex) {
        // Convert to COO first for more efficient modification.
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Copies this matrix and sets the specified index of the copy to {@code value}.
     * @param value Value to set.
     * @param row Row index of value to be set.
     * @param col Column index of value to be set.
     * @return A copy of this matrix with the specified index set to {@code value}.
     */
    public CsrCMatrix set(double value, int row, int col) {
        return set(new Complex128(value), row, col);
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
    public Complex128 get(int... indices) {
        ValidateParameters.ensureValidIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];
        int loc = Arrays.binarySearch(colIndices, rowPointers[row], rowPointers[row+1], col);

        if(loc >= 0) return (Complex128) entries[loc];
        else return Complex128.ZERO;
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CsrMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2CSR(CsrCMatrix B) {
        return ComplexCsrMatrixMultiplication.standardAsSparse(this, B);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CsrMatrix)} if the result of multiplying this matrix
     * with {@code b} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param b Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code b} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2CSR(CsrMatrix b) {
        return RealComplexCsrMatrixMultiplication.standardAsSparse(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix does not equal the number of rows in matrix {@code b}.
     */
    public CMatrix mult(CsrMatrix b) {
        return RealComplexCsrMatrixMultiplication.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix does not equal the number of rows in matrix {@code b}.
     */
    public CMatrix mult(CMatrix B) {
        ValidateParameters.ensureMatMultShapes(shape, B.shape);

        return new CMatrix(numRows, B.numCols,
                ComplexDenseSparseMatrixMultiplication.standard(
                        entries, rowPointers, colIndices, shape,
                        B.entries, B.shape
                )
        );
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix does not equal the number of rows in matrix {@code b}.
     */
    public CMatrix mult(Matrix b) {
        return RealComplexCsrDenseMatrixMultiplication.standard(this, b);
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
        result = 31*result + shape.hashCode();

        // Hash calculation ignores explicit zeros in the matrix. This upholds the contract with the equals(Object) method.
        for(int row = 0; row<numRows; row++) {
            for(int idx = rowPointers[row], rowStop = rowPointers[row + 1]; idx < rowStop; idx++) {
                if (!entries[idx].isZero()) {
                    result = 31 * result + entries[idx].hashCode();
                    result = 31 * result + Integer.hashCode(colIndices[idx]);
                    result = 31 * result + Integer.hashCode(row);
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
