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

package org.flag4j.core_temp.arrays.sparse;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.MatrixMixin;
import org.flag4j.core_temp.MatrixVectorOpsMixin;
import org.flag4j.core_temp.PrimitiveDoubleTensorBase;
import org.flag4j.core_temp.arrays.dense.Matrix;
import org.flag4j.core_temp.arrays.dense.Vector;
import org.flag4j.operations.sparse.csr.real.RealCsrManipulations;
import org.flag4j.operations.sparse.csr.real.RealCsrMatrixMultiplication;
import org.flag4j.operations.sparse.csr.real.RealCsrOperations;
import org.flag4j.operations.sparse.csr.real.RealCsrProperties;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;

import static org.flag4j.operations.sparse.coo.SparseUtils.sortCsrMatrix;


/**
 * <p>A real sparse matrix stored in compressed sparse row (CSR) format. The {@link #entries} of this CSR matrix are
 * primitive doubles.</p>
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
 * <p>If indices need to be sorted, call {@link #sortIndices()}.</p>
 */
public class CsrMatrix extends PrimitiveDoubleTensorBase<CsrMatrix, Matrix>
        implements CsrMatrixMixin<CsrMatrix, Matrix, Double>, MatrixVectorOpsMixin<CsrMatrix, CooVector, Vector> {


    /**
     * <p>Pointers indicating starting index of each row within the {@link #colIndices} and {@link #entries} arrays.
     * Has length {@link #numRows numRows + 1}.</p>
     *
     * <p>The range {@code [entries[rowPointers[i]], entries[rowPointers[i+1]])} contains all {@link #entries non-zero entries} within
     * row {@code i}.</p>
     *
     * <p>Similarly, {@code [colIndices[rowPointers[i]], colIndices[rowPointers[i+1]])} contains all {@link #colIndices column indices}
     * for the entries in row {@code i}.
     * </p>
     */
    public final int[] rowPointers;
    /**
     * Column indices for non-zero values of this sparse CSR matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero entries in this CSR matrix.
     */
    public final int nnz;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;
    /**
     * The sparsity of this matrix.
     */
    private double sparsity = -1;


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
    public CsrMatrix(Shape shape, double[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries);
        ParameterChecks.ensureRank(shape, 2);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.nnz = entries.length;
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column indices.
     *
     * @param numRows The number of rows in this matrix.
     * @param numCols The number of columns in this matrix.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.</p>
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    public CsrMatrix(int numRows, int numCols, double[] entries, int[] rowPointers, int[] colIndices) {
        super(new Shape(numRows, numCols), entries);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.nnz = entries.length;
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
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
    public Matrix tensorDot(CsrMatrix src2, int[] aAxes, int[] bAxes) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Constructs a CSR matrix of the same type as this matrix with the given the {@code shape} and {@code entries} and the same
     * row pointers and column indices as this matrix.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A CSR matrix of the same type as this matrix with the given the {@code shape} and {@code entries} and the same
     * row pointers and column indices as this matrix.
     */
    @Override
    public CsrMatrix makeLikeTensor(Shape shape, double[] entries) {
        return new CsrMatrix(shape, entries, rowPointers, colIndices);
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
    public CsrMatrix T(int axis1, int axis2) {
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return copy();

        return RealCsrOperations.transpose(this);
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
    public CsrMatrix T(int... axes) {
        if(axes.length != 2 || !((axes[0] == 0 && axes[1] == 1) || (axes[0] == 1 && axes[1] == 0))) {
            throw new LinearAlgebraException("Cannot transpose axes: "  + Arrays.toString(axes) + " for tensor of rank 2.");
        }

        return RealCsrOperations.transpose(this);
    }


    /**
     * The sparsity of this sparse CSR matrix. That is, the decimal percentage of elements in this matrix which are zero.
     *
     * @return The density of this sparse matrix.
     */
    @Override
    public double sparsity() {
        BigDecimal sparsity = new BigDecimal(this.totalEntries()).subtract(BigDecimal.valueOf(this.nnz));
        sparsity = sparsity.divide(new BigDecimal(this.totalEntries()), 50, RoundingMode.HALF_UP);

        return sparsity.doubleValue();
    }


    /**
     * Converts this sparse CSR matrix to an equivalent dense matrix.
     *
     * @return A dense matrix equivalent to this sparse CSR matrix.
     */
    @Override
    public Matrix toDense() {
        double[] dest = new double[shape.totalEntries().intValueExact()];

        for(int i=0; i<rowPointers.length-1; i++) {
            int rowOffset = i*numCols;

            for(int j=rowPointers[i]; j<rowPointers[i+1]; j++) {
                dest[rowOffset + colIndices[j]] = entries[j];
            }
        }

        return new Matrix(shape, dest);
    }


    /**
     * Converts this CSR matrix to an equivalent {@link CooMatrix COO matrix}.
     * @return A {@link CooMatrix COO matrix} equivalent to this matrix.
     */
    public CooMatrix toCoo() {
        int[] destRowIdx = new int[entries.length];

        for(int i=0; i<numRows; i++) {
            for(int j=rowPointers[i], stop=rowPointers[i+1]; j<stop; j++)
                destRowIdx[j] = i;
        }

        return new CooMatrix(shape, entries.clone(), destRowIdx, colIndices.clone());
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    @Override
    public void sortIndices() {
        sortCsrMatrix(entries, rowPointers, colIndices);
    }


    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    @Override
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     *
     * @return The number of columns in this matrix.
     */
    @Override
    public int numCols() {
        return numCols;
    }


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.</p>
     *
     * <p>Same as {@link #trace()}.</p>
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double tr() {
        ParameterChecks.ensureSquareMatrix(shape);
        double trace = 0;

        for(int i=0; i<numRows; i++) {
            int rowPtr = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowPtr; j<stop; j++) {
                if(i==colIndices[j]) {
                    trace += entries[j];
                }
            }
        }

        return trace;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     *
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        if(!isSquare()) return false;

        for(int i=1; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowStart; j<stop; j++) {
                if(colIndices[j] >= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(entries[j] != 0) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point then the matrix must be upper triangular.
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     *
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        if(!isSquare()) return false;

        for(int i=0; i<numRows; i++) {
            int rowStart = rowPointers[i];
            int rowStop = rowPointers[i+1];

            for(int j=rowStop-1; j>=rowStart; j--) {
                if(colIndices[j] <= i) break; // Have reached the diagonal. No need to continue for this row.
                else if(entries[j] != 0) return false; // Non-zero entry found. No need to continue.
            }
        }

        return true; // If we reach this point then the matrix must be lower-triangular.
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     *
     * @see #isCloseToI()
     */
    @Override
    public boolean isI() {
        return RealCsrProperties.isIdentity(this);
    }


    /**
     * Checks that this matrix is close to the identity matrix.
     *
     * @return True if this matrix is approximately the identity matrix.
     *
     * @see #isI()
     */
    @Override
    public boolean isCloseToI() {
        return RealCsrProperties.isCloseToIdentity(this);
    }


    /**
     * <p>Computes the determinant of a square matrix.</p>
     * <p><b>WARNING:</b> This method will convert the matrix to a dense matrix in order to compute the determinant.</p>
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    @Override
    public Double det() {
        return toDense().det();
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of rows in matrix {@code b}.
     */
    @Override
    public Matrix mult(CsrMatrix b) {
        return RealCsrMatrixMultiplication.standard(this, b);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public Matrix multTranspose(CsrMatrix b) {
        return RealCsrMatrixMultiplication.standard(this, b.T());
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix b.
     *
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    @Override
    public Double fib(CsrMatrix b) {
        return this.T().mult(b).tr();
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(MatrixMixin, int) 
     * @see #augment(CsrMatrix) 
     */
    @Override
    public CsrMatrix stack(CsrMatrix b) {
        return toCoo().stack(b.toCoo()).toCsr();
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(T)
     * @see #stack(T, int)
     */
    @Override
    public CsrMatrix augment(CsrMatrix b) {
        return toCoo().augment(b.toCoo()).toCsr();
    }


    /**
     * Swaps specified rows in the matrix. This is done in place.
     *
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public CsrMatrix swapRows(int rowIndex1, int rowIndex2) {
        RealCsrManipulations.swapRows(this, rowIndex1, rowIndex2);
        return this;
    }


    /**
     * Swaps specified columns in the matrix. This is done in place.
     *
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public CsrMatrix swapCols(int colIndex1, int colIndex2) {
        RealCsrManipulations.swapRows(this, colIndex1, colIndex2);
        return this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     *
     * @see #isAntiSymmetric()
     */
    @Override
    public boolean isSymmetric() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     *
     * @see #isSymmetric()
     */
    @Override
    public boolean isAntiSymmetric() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        // TODO: Implementation.
        return false;
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public CsrMatrix removeRow(int rowIndex) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeRows(int... rowIndices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeCol(int colIndex) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeCols(int... colIndices) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A copy of this matrix with the given slice set to the specified values.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSliceCopy(CsrMatrix values, int rowStart, int colStart) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     *
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public CsrMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     *
     * @return A reference to this matrix.
     */
    @Override
    public CsrMatrix set(Double value, int row, int col) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Extracts the upper-triangular portion of this matrix with a specified diagonal offset. All other entries of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for upper-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all entries at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all entries at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all entries at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The upper-triangular portion of this matrix with a specified diagonal offset. All other entries of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public CsrMatrix getTriU(int diagOffset) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Extracts the lower-triangular portion of this matrix with a specified diagonal offset. All other entries of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for lower-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all entries at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all entries at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all entries at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The lower-triangular portion of this matrix with a specified diagonal offset. All other entries of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public CsrMatrix getTriL(int diagOffset) {
        // TODO: Implementation.
        return null;
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
    public Vector mult(CooVector b) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooVector toVector() {
        // TODO: Implementation.
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
    public CooVector getRow(int rowIdx) {
        // TODO: Implementation.
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
    public CooVector getRow(int rowIdx, int colStart, int colEnd) {
        // TODO: Implementation.
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
    public CooVector getCol(int colIdx) {
        // TODO: Implementation.
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
    public CooVector getCol(int colIdx, int rowStart, int rowEnd) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public CooVector getDiag() {
        // TODO: Implementation.
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
    public CsrMatrix setCol(CooVector values, int colIndex) {
        // TODO: Implementation.
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
    public CsrMatrix setRow(CooVector values, int rowIndex) {
        // TODO: Implementation.
        return null;
    }
}
