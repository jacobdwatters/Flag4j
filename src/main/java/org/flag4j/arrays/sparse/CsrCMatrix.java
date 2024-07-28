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

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.ComplexMatrixMixin;
import org.flag4j.core.MatrixMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.sparse_base.ComplexSparseTensorBase;
import org.flag4j.io.PrintOptions;
import org.flag4j.operations.dense_sparse.csr.complex.ComplexCsrDenseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.complex.ComplexCsrDenseOperations;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseOperations;
import org.flag4j.operations.sparse.csr.complex.*;
import org.flag4j.operations.sparse.csr.real_complex.RealComplexCsrMatrixMultiplication;
import org.flag4j.operations.sparse.csr.real_complex.RealComplexCsrOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>Complex sparse matrix stored in compressed sparse row (CSR) format.</p>
 *
 * <p>CSR matrices are best suited for efficient access and matrix operations. Specifically, matrix-matrix and
 * matrix-vector multiplication. CSR matrices are <b>not</b> well suited for modification (see {@link CooCMatrix}).</p>
 *
 * <p>The CSR format stores a sparse <code>m-by-n</code> matrix as three one-dimensional arrays: {@link #entries},
 * {@link #rowPointers}, and {@link #colIndices}.</p>
 *
 * <ul>
 *     <li><b>entries</b>: Stores the non-zero values of the sparse matrix. Note, zero values can be stored explicitly in this array.
 *     Hence, the term "non-zero values" is a misnomer.</li>
 *     <li><b>rowPointers</b>: Encodes the total number of non-zero values above each row. Has length <code>m+1</code>. For example,
 *     <code>rowPointers[j]</code> contains the total number of non-zero values above row <code>j</code>. The first entry is always
 *     0 and the last element is always <code>entries.length</code></li>
 *     <li><b>colIndices</b>: Contains the column indices for all non-zero entries. Has length <code>entries.length</code></li>
 * </ul>
 * @see CooCMatrix
 * @see CsrMatrix
 */
public class CsrCMatrix
        extends ComplexSparseTensorBase<CsrCMatrix, CMatrix, CsrMatrix>
        implements MatrixMixin<CsrCMatrix, CMatrix, CsrCMatrix, CsrCMatrix, CNumber, CooCVector, CVector>,
        ComplexMatrixMixin<CsrCMatrix> {

    /**
     * Row indices of the non-zero entries of the sparse matrix.
     */
    public final int[] rowPointers;
    /**
     * Column indices of the non-zero entries of the sparse matrix.
     */
    public final int[] colIndices;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;
    /**
     * The number of non-zero entries stored in this sparse matrix.
     */
    public final int nnz;


    /**
     * Constructs an empty sparse CSR matrix with the specified shape.
     * @param shape Shape of the CSR matrix.
     */
    public CsrCMatrix(Shape shape) {
        super(shape, 0, new CNumber[0], new int[shape.dims[0]+1], new int[0]);

        numRows = shape.dims[0];
        numCols = shape.dims[1];
        this.rowPointers = indices[0];
        this.colIndices = indices[1];
        nnz = entries.length;
    }


    /**
     * Constructs an empty sparse CSR matrix with the specified shape.
     * @param numRows Number of rows in the CSR matrix.
     * @param numCols Number of columns in the CSR matrix.
     */
    public CsrCMatrix(int numRows, int numCols) {
        super(new Shape(numRows, numCols), 0, new CNumber[0], new int[numRows+1], new int[0]);

        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        this.rowPointers = indices[0];
        this.colIndices = indices[1];
        nnz = entries.length;
    }


    /**
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrCMatrix(Shape shape, CNumber[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries.length, entries, rowPointers, colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
        nnz = entries.length;
    }


    /**
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param numRows Number of rows for CSR matrix.
     * @param numCols Number of columns for CSR matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrCMatrix(int numRows, int numCols, CNumber[] entries, int[] rowPointers, int[] colIndices) {
        super(new Shape(numRows, numCols), entries.length, entries, rowPointers, colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        nnz = entries.length;
    }


    /**
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrCMatrix(Shape shape, double[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries.length, new CNumber[entries.length], new int[colIndices.length], colIndices);

        ArrayUtils.copy2CNumber(entries, this.entries); // Copy entries from double array to CNumber array.

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
        nnz = entries.length;
    }


    /**
     * Constructs a sparse CSR matrix which is a deep copy of the {@code src} matrix.
     * @param src Matrix to create copy of.
     */
    public CsrCMatrix(CsrCMatrix src) {
        super(src.shape.copy(), src.entries.length, ArrayUtils.copyOf(src.entries),
                src.rowPointers.clone(), src.colIndices.clone());

        this.rowPointers = indices[0];
        this.colIndices = indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        nnz = entries.length;
    }


    /**
     * Converts a sparse COO matrix to a sparse CSR matrix.
     * @param src COO matrix to convert. Indices must be sorted lexicographically.
     */
    public CsrCMatrix(CooCMatrix src) {
        super(src.shape.copy(),
                src.entries.length,
                new CNumber[src.entries.length],
                new int[src.numRows + 1],
                src.colIndices.clone()
        );

        rowPointers = this.indices[0];
        colIndices = this.indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        nnz = entries.length;

        ArrayUtils.copy2CNumber(src.entries, entries); // Deep copy non-zero entries.

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0; i<src.entries.length; i++) {
            rowPointers[src.rowIndices[i] + 1]++;
        }

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<src.numRows; i++) {
            rowPointers[i+1] += rowPointers[i];
        }
    }


    /**
     * Converts a real sparse COO matrix to a complex sparse CSR matrix.
     * @param src COO matrix to convert. Indices must be sorted lexicographically.
     */
    public CsrCMatrix(CooMatrix src) {
        super(src.shape.copy(),
                src.entries.length,
                new CNumber[src.entries.length],
                new int[src.numRows + 1],
                src.colIndices.clone()
        );

        rowPointers = this.indices[0];
        colIndices = this.indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        nnz = entries.length;

        ArrayUtils.copy2CNumber(src.entries, entries); // Deep copy non-zero entries.

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0; i<src.entries.length; i++) {
            rowPointers[src.rowIndices[i] + 1]++;
        }

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<src.numRows; i++) {
            rowPointers[i+1] += rowPointers[i];
        }
    }


    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CsrCMatrix H() {
        return ComplexCsrOperations.hermTranspose(this);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return ComplexCsrProperties.isHermitian(this);
    }


    /**
     * Checks if a matrix is anti-Hermitian. That is, if the matrix is equal to the negative of its conjugate transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isAntiHermitian() {
        return ComplexCsrProperties.isAntiHermitian(this);
    }


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is equal to its conjugate transpose.
     *
     * @return True if this matrix it is unitary. Otherwise, returns false.
     */
    @Override
    public boolean isUnitary() {
        return isSquare() && this.mult(this.H()).allClose(CMatrix.I(numRows));
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException  If the values vector has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex} is not within the matrix.
     */
    @Override
    public CsrCMatrix setCol(CVector values, int colIndex) {
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException  If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    @Override
    public CsrCMatrix setRow(CVector values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException  If the {@code values} vector has a different length than the number of columns of this matrix.
     * @throws IndexOutOfBoundsException If {@code rowIndex} is not within the matrix.
     */
    @Override
    public CsrCMatrix setRow(CooCVector values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(CooCMatrix values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException  If the number of indices is not equal to the rank of this tensor.
     * @throws IndexOutOfBoundsException If any of the indices are not within this tensor.
     */
    @Override
    public CsrCMatrix set(CNumber value, int... indices) {
        ParameterChecks.assertEquals(2, indices.length);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CsrCMatrix getSelf() {
        return this;
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     *
     * @param tensor Tensor to compare this tensor to.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (atol + rtol*|b|)}. Otherwise, returns false.
     * @see #allClose(Object)
     */
    @Override
    public boolean allClose(CsrCMatrix tensor, double relTol, double absTol) {
        return ComplexCsrEquals.allClose(this, tensor, relTol, absTol);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public CsrCMatrix set(double value, int... indices) {
        ParameterChecks.assertEquals(indices.length, 2);
        return set(new CNumber(value), indices[0], indices[1]);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public CsrCMatrix flatten(int axis) {
        if(axis==0) {
            // Flatten to single row.
            return reshape(new Shape(1, entries.length));
        } else if(axis==1) {
            // Flatten to single column.
            return reshape(new Shape(entries.length, 1));
        } else {
            // Unknown axis.
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
        }
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CsrCMatrix add(CsrCMatrix B) {
        return ComplexCsrOperations.applyBinOpp(B, this, CNumber::add, null);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    public CsrCMatrix add(CsrMatrix B) {
        return RealComplexCsrOperations.applyBinOpp(B, this, (Double a, CNumber b)->b.add(a), null);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(double a) {
        return ComplexCsrDenseOperations.applyBinOpp(this, a, CNumber::add, null);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        return ComplexCsrDenseOperations.applyBinOpp(this, a, CNumber::add, null);
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CsrCMatrix sub(CsrCMatrix B) {
        return ComplexCsrOperations.applyBinOpp(this, B, CNumber::add, CNumber::addInv);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix sub(double a) {
        return ComplexCsrDenseOperations.applyBinOpp(this, a, CNumber::add, (Double x) -> -x);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        return ComplexCsrDenseOperations.applyBinOpp(this, a, CNumber::add, CNumber::addInv);
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     * @see #H()
     * @see #hermTranspose()
     */
    @Override
    public CsrCMatrix transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     * @see #H()
     * @see #hermTranspose()
     */
    @Override
    public CsrCMatrix T() {
        return ComplexCsrOperations.transpose(this);
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
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public CNumber get(int... indices) {
        ParameterChecks.assertValidIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];
        int loc = Arrays.binarySearch(colIndices, rowPointers[row], rowPointers[row+1], col);

        if(loc >= 0) return entries[loc];
        else return CNumber.zero();
    }


    /**
     * Gets the shape of this matrix.
     *
     * @return The shape of this matrix.
     */
    @Override
    public Shape shape() {
        return shape;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CsrCMatrix copy() {
        return new CsrCMatrix(this);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CsrCMatrix elemMult(CsrCMatrix B) {
        return ComplexCsrOperations.applyBinOpp(this,B, CNumber::mult, null);
    }


    /**
     * Computes the element-wise addition between two matrices.
     *
     * @param B Second matrix in the addition.
     *
     * @return The result of adding the matrix B to this matrix element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(Matrix B) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, B, CNumber::add, null);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CsrCMatrix add(CooMatrix B) {
        return this.add(B.toCsr());
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(CMatrix B) {
        return ComplexCsrDenseOperations.applyBinOpp(this, B, CNumber::add, null);
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     *
     * @return The result of adding the tensor B to this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CsrCMatrix add(CooCMatrix B) {
        return this.add(B.toCsr());
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(Matrix B) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, B, CNumber::add, (Double b)->-b);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CsrCMatrix sub(CooMatrix B) {
        return this.sub(B.toCsr());
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public CsrCMatrix sub(CsrMatrix B) {
        return RealComplexCsrOperations.applyBinOpp(this, B, CNumber::add, (Double b)->-b);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(CMatrix B) {
        return ComplexCsrDenseOperations.applyBinOpp(this, B, CNumber::add, CNumber::addInv);
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     *
     * @return The result of subtracting the tensor B from this tensor element-wise.
     *
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CsrCMatrix sub(CooCMatrix B) {
        return this.sub(B.toCsr());
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CsrMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2CSR(CsrMatrix B) {
        return RealComplexCsrMatrixMultiplication.standardAsSparse(this, B);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CsrCMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2CSR(CsrCMatrix B) {
        return ComplexCsrMatrixMultiplication.standardAsSparse(this, B);
    }


    /**
     * Computes the matrix multiplication between two sparse matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CooMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2CSR(CooMatrix B) {
        return RealComplexCsrMatrixMultiplication.standardAsSparse(this, B.toCsr());
    }


    /**
     * Computes the matrix multiplication between two sparse matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CooCMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrCMatrix mult2CSR(CooCMatrix B) {
        return ComplexCsrMatrixMultiplication.standardAsSparse(this, B.toCsr());
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  rows in matrix {@code B}.
     */
    @Override
    public CMatrix mult(Matrix B) {
        return RealComplexCsrDenseMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix-vector multiplication.
     *
     * @param b Vector to multiply this matrix to.
     *
     * @return The vector result from multiplying this matrix by the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  entries {@code b}.
     */
    @Override
    public CVector mult(CooCVector b) {
        return ComplexCsrMatrixMultiplication.standardVector(this, b);
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method will be significantly
     * faster.
     *
     * @param exponent The exponent in the matrix power.
     *
     * @return The result of multiplying this matrix with itself 'exponent' times.
     */
    @Override
    public CMatrix pow(int exponent) {
        ParameterChecks.assertPositive(exponent);

        if(exponent==0) {
            return CMatrix.I(shape);
        } else if(exponent==1) {
            return this.toDense();
        }
        else {
            CMatrix exp = this.mult(this); // First multiplication is sparse-sparse multiplication.

            for(int i=2; i<exponent; i++)
                exp = exp.mult(this);

            return exp;
        }
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     *
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CsrCMatrix elemMult(Matrix B) {
        return RealComplexCsrDenseOperations.applyBinOppToSparse(B, this, (Double a, CNumber b)  -> b.mult(a));
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     *
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CsrCMatrix elemMult(CooMatrix B) {
        return this.elemMult(B.toCsr());
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     *
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public CsrCMatrix elemMult(CsrMatrix B) {
        return RealComplexCsrOperations.applyBinOpp(this, B, CNumber::mult, null);
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     *
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CsrCMatrix elemMult(CMatrix B) {
        return ComplexCsrDenseOperations.applyBinOppToSparse(B, this, CNumber::mult);
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     *
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CooCMatrix elemMult(CooCMatrix B) {
        // TODO: This should return a CsrCMatrix. Need to add complex sparse type as generic type parameter to MatrixOperationsMixin
        //  for this to work properly.
        return this.elemMult(B.toCsr()).toCoo();
    }


    /**
     * Computes the element-wise division between two matrices.
     *
     * @param B Second matrix in the element-wise division.
     *
     * @return The result of element-wise division of this matrix with the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException      If B contains any zero entries.
     */
    @Override
    public CsrCMatrix elemDiv(Matrix B) {
        return RealComplexCsrDenseOperations.applyBinOppToSparse(B, this, (Double x, CNumber y)->y.div(x));
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  rows in matrix {@code B}.
     */
    @Override
    public CMatrix mult(CsrCMatrix B) {
        return ComplexCsrMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CooMatrix B) {
        return mult(B.toCsr());
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    public CMatrix mult(CsrMatrix B) {
        return RealComplexCsrMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CMatrix B) {
        return ComplexCsrDenseMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CooCMatrix B) {
        return this.mult(B.toCsr());
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(Vector b) {
        return RealComplexCsrDenseMatrixMultiplication.standardVector(this, b);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(CooVector b) {
        return RealComplexCsrMatrixMultiplication.standardVector(this, b);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(CVector b) {
        return ComplexCsrDenseMatrixMultiplication.standardVector(this, b);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CsrCMatrix elemDiv(CMatrix B) {
        return ComplexCsrDenseOperations.applyBinOppToSparse(B, this, (CNumber a, CNumber b)->b.div(a));
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public CNumber det() {
        return toDense().det();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(Matrix B) {
        return this.H().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CooMatrix B) {
        return this.H().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CMatrix B) {
        return this.H().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CooCMatrix B) {
        return this.H().mult(B).tr();
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vectors of length m.
     */
    @Override
    public CVector sumCols() {
        CVector sum = new CVector(numRows);

        int rowStop = rowPointers.length-1;
        for(int i=0; i<rowStop; i++) {
            for(int j=rowPointers[i]; j<rowPointers[i+1]; j++) {
                sum.entries[i].addEq(entries[j]);
            }
        }

        return sum;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vector of length n.
     */
    @Override
    public CVector sumRows() {
        CVector sum = new CVector(numCols);

        int nnz = entries.length;
        for(int i=0; i<nnz; i++) {
            sum.entries[colIndices[i]].addEq(entries[i]);
        }

        return new CVector(sum);
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     *
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(Vector b) {
        return ComplexCsrOperations.addToEachCol(this, b);
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     *
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CooVector b) {
        return ComplexCsrOperations.addToEachCol(this, b);
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     *
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CVector b) {
        return ComplexCsrOperations.addToEachCol(this, b);
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     *
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CooCVector b) {
        return ComplexCsrOperations.addToEachCol(this, b);
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     *
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(Vector b) {
        return ComplexCsrOperations.addToEachRow(this, b);
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     *
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CooVector b) {
        return ComplexCsrOperations.addToEachRow(this, b);
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     *
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CVector b) {
        return ComplexCsrOperations.addToEachRow(this, b);
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     *
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CooCVector b) {
        return ComplexCsrOperations.addToEachRow(this, b);
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(Matrix B) {
        return toCoo().stack(B);
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CsrCMatrix stack(CooMatrix B) {
        return toCoo().stack(B).toCsr();
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(CMatrix B) {
        return toCoo().stack(B);
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix B.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CsrCMatrix stack(CooCMatrix B) {
        return toCoo().stack(B).toCsr();
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking B to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(Matrix B) {
        return toCoo().augment(B);
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking B to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CsrCMatrix augment(CooMatrix B) {
        return toCoo().augment(B).toCsr();
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking B to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(CMatrix B) {
        return toCoo().augment(B);
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     *
     * @return The result of stacking B to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CsrCMatrix augment(CooCMatrix B) {
        return toCoo().augment(B).toCsr();
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(Vector, int)} and {@link #augment(Vector)}.
     *
     * @param b Vector to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrCMatrix stack(Vector b) {
        return toCoo().stack(b).toCsr();
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooVector, int)} and {@link #augment(CooVector)}.
     *
     * @param b Vector to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrCMatrix stack(CooVector b) {
        return toCoo().stack(b).toCsr();
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CVector, int)} and {@link #augment(CVector)}.
     *
     * @param b Vector to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrCMatrix stack(CVector b) {
        return toCoo().stack(b).toCsr();
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooCVector, int)} and {@link #augment(CooCVector)}.
     *
     * @param b Vector to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the vector b.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrCMatrix stack(CooCVector b) {
        return toCoo().stack(b).toCsr();
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(Vector)} and {@link #stack(Vector, int)}.
     *
     * @param b vector to augment to this matrix.
     *
     * @return The result of augmenting b to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrCMatrix augment(Vector b) {
        return toCoo().augment(b).toCsr();
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooVector)} and {@link #stack(CooVector, int)}.
     *
     * @param b vector to augment to this matrix.
     *
     * @return The result of augmenting b to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrCMatrix augment(CooVector b) {
        return toCoo().augment(b).toCsr();
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CVector)} and {@link #stack(CVector, int)}.
     *
     * @param b vector to augment to this matrix.
     *
     * @return The result of augmenting b to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrCMatrix augment(CVector b) {
        return toCoo().augment(b).toCsr();
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooCVector)} and {@link #stack(CooCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     *
     * @return The result of augmenting b to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrCMatrix augment(CooCVector b) {
        return toCoo().augment(b).toCsr();
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     *
     * @return The specified row of this matrix.
     */
    @Override
    public CooCVector getRow(int i) {
        int start = rowPointers[i];

        CNumber[] destEntries = new CNumber[rowPointers[i+1]-start];
        int[] destIndices = new int[destEntries.length];

        ArrayUtils.arraycopy(entries, start, destEntries, 0, destEntries.length);
        System.arraycopy(colIndices, start, destIndices, 0, destEntries.length);

        return new CooCVector(this.numCols, destEntries, destIndices);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     *
     * @return The specified column of this matrix.
     */
    @Override
    public CooCVector getCol(int j) {
        return getCol(j, 0, numRows);
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
     * @throws IllegalArgumentException   If {@code rowStart} is less than 0.
     * @throws NegativeArraySizeException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public CooCVector getCol(int colIdx, int rowStart, int rowEnd) {
        List<CNumber> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=rowStart; i<rowEnd; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                if(colIndices[j]==colIdx) {
                    destEntries.add(entries[j].copy());
                    destIndices.add(i);
                    break; // Should only be a single entry with this row and column index.
                }
            }
        }

        return new CooCVector(numRows, destEntries, destIndices);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooCVector toVector() {
        int type = vectorType();

        CNumber[] destEntries = ArrayUtils.copyOf(entries); // Copy non-zero values.
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

        return new CooCVector(shape.totalEntries().intValueExact(), destEntries, indices);
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
    public CsrCMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        return ComplexCsrOperations.getSlice(this, rowStart, rowEnd, colStart, colEnd);
    }


    /**
     * Get a specified column of this matrix at and below a specified row.
     *
     * @param rowStart Index of the row to begin at.
     * @param j Index of column to get.
     *
     * @return The specified column of this matrix beginning at the specified row.
     *
     * @throws NegativeArraySizeException     If {@code rowStart} is larger than the number of rows in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowStart} or {@code j} is outside the bounds of this matrix.
     */
    @Override
    public CooCVector getColBelow(int rowStart, int j) {
        return getCol(j, rowStart, numRows);
    }


    /**
     * Get a specified row of this matrix at and after a specified column.
     *
     * @param colStart Index of the row to begin at.
     * @param rowIdx Index of the row to get.
     *
     * @return The specified row of this matrix beginning at the specified column.
     *
     * @throws NegativeArraySizeException     If {@code colStart} is larger than the number of columns in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} or {@code colStart} is outside the bounds of this matrix.
     */
    @Override
    public CooCVector getRowAfter(int colStart, int rowIdx) {
        int start = rowPointers[rowIdx];
        int end = rowPointers[rowIdx+1];

        List<CNumber> row = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int j=start; j<end; j++) {
            int col = colIndices[j];

            if(col >= colStart) {
                row.add(entries[j].copy());
                indices.add(col-colStart);
            }
        }

        return new CooCVector(this.numCols-colStart, row, indices);
    }


    /**
     * Sets a column of this matrix.
     *
     * @param values Vector containing the new values for the matrix.
     * @param j Index of the column of this matrix to set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException  If the number of entries in the {@code values} vector
     *                                   is not the same as the number of rows in this matrix.
     * @throws IndexOutOfBoundsException If {@code j} is not within the bounds of this matrix.
     */
    @Override
    public CsrCMatrix setCol(CooCVector values, int j) {
        return toCoo().setCol(values, j).toCsr();
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #tr()}.
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public CNumber trace() {
        return tr();
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public CNumber tr() {
        ParameterChecks.assertSquareMatrix(shape);

        CNumber trace = new CNumber();

        for(int i=0; i<numRows; i++) {
            int rowPtr = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowPtr; j<stop; j++) {
                if(i==colIndices[j]) {
                    trace.addEq(entries[j]);
                }
            }
        }

        return trace;
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public CooCVector getDiag() {
        List<CNumber> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<numRows; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            int loc = Arrays.binarySearch(colIndices, start, stop, i); // Search for matching column index

            if(loc >= 0) {
                destEntries.add(entries[loc].copy());
                destIndices.add(i);
            }
        }

        return new CooCVector(Math.min(numRows, numCols), destEntries, destIndices);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CsrCMatrix makeTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CsrCMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * A factory for creating a real sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CsrMatrix makeRealTensor(Shape shape, double[] entries, int[][] indices) {
        return new CsrMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public CMatrix toDense() {
        CNumber[] dest = new CNumber[shape.totalEntries().intValueExact()];
        ArrayUtils.fillZeros(dest);

        for(int i=0; i<rowPointers.length-1; i++) {
            int rowOffset = i*numCols;

            for(int j=rowPointers[i]; j<rowPointers[i+1]; j++) {
                dest[rowOffset + colIndices[j]] = entries[j].copy();
            }
        }

        return new CMatrix(shape.copy(), dest);
    }


    /**
     * Converts this matrix to an equivalent sparse tensor.
     * @return A sparse tensor which is equivalent to this matrix.
     */
    public CooCTensor toTensor() {
        return toCoo().toTensor();
    }


    /**
     * Converts this {@link CsrMatrix CSR matrix} to an equivalent {@link CooMatrix COO matrix}.
     * @return A {@link CooMatrix COO matrix} equivalent to this {@link CsrMatrix CSR matrix}.
     */
    public CooCMatrix toCoo() {
        CNumber[] dest = entries.clone();
        int[] destRowIdx = new int[entries.length];
        int[] destColIdx = colIndices.clone();

        for(int i=0; i<numRows; i++) {
            int stop = rowPointers[i+1];

            for(int j=rowPointers[i]; j<stop; j++) {
                destRowIdx[j] = i;
            }
        }

        return new CooCMatrix(shape.copy(), dest, destRowIdx, destColIdx);
    }


    /**
     * Checks if this matrix is the identity matrix.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        return ComplexCsrProperties.isIdentity(this);
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
    public CsrCMatrix set(double value, int row, int col) {
        return set(new CNumber(value), row, col);
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
    public CsrCMatrix set(CNumber value, int row, int col) {
        // Ensure indices are in bounds.
        ParameterChecks.assertValidIndex(shape, row, col);

        CNumber[] newEntries;
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
            newEntries = ArrayUtils.copyOf(entries);
            newEntries[loc] = value.copy();
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch
            newEntries = new CNumber[entries.length + 1];
            newColIndices = new int[entries.length + 1];

            // Copy old entries and insert new one.
            ArrayUtils.arraycopy(entries, 0, newEntries, 0, loc);
            newEntries[loc] = value.copy();
            ArrayUtils.arraycopy(entries, loc, newEntries, loc+1, entries.length-loc);

            // Copy old column indices and insert new one.
            System.arraycopy(colIndices, 0, newColIndices, 0, loc);
            newColIndices[loc] = col;
            System.arraycopy(colIndices, loc, newColIndices, loc+1, entries.length-loc);

            // Increment row pointers.
            for(int i=row+1; i<rowPointers.length; i++) {
                newRowPointers[i]++;
            }
        }

        return new CsrCMatrix(shape.copy(), newEntries, newRowPointers, newColIndices);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrCMatrix setCol(CNumber[] values, int colIndex) {
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrCMatrix setCol(Double[] values, int colIndex) {
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrCMatrix setCol(Integer[] values, int colIndex) {
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrCMatrix setCol(double[] values, int colIndex) {
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrCMatrix setCol(int[] values, int colIndex) {
        return toCoo().setCol(values, colIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrCMatrix setRow(CNumber[] values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrCMatrix setRow(Double[] values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrCMatrix setRow(Integer[] values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrCMatrix setRow(double[] values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrCMatrix setRow(int[] values, int rowIndex) {
        return toCoo().setRow(values, rowIndex).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(CsrCMatrix values, int rowStart, int colStart) {
        return toCoo().setSlice(values.toCoo(), rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(Matrix values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(CooMatrix values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(CNumber[][] values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(Double[][] values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(Integer[][] values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(double[][] values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrCMatrix setSlice(int[][] values, int rowStart, int colStart) {
        return toCoo().setSlice(values, rowStart, colStart).toCsr();
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return a copy of this matrix with the specified row removed.
     */
    @Override
    public CsrCMatrix removeRow(int rowIndex) {
        return toCoo().removeRow(rowIndex).toCsr();
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix.
     *
     * @return a copy of this matrix with the specified rows removed.
     */
    @Override
    public CsrCMatrix removeRows(int... rowIndices) {
        return toCoo().removeRows(rowIndices).toCsr();
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrCMatrix removeCol(int colIndex) {
        return toCoo().removeCol(colIndex).toCsr();
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     *
     * @return a copy of this matrix with the specified columns removed.
     */
    @Override
    public CsrCMatrix removeCols(int... colIndices) {
        return toCoo().removeCols(colIndices).toCsr();
    }



    /**
     * Swaps rows in the matrix.
     *
     * @param rowIndex1 Index of first row to swap.
     * @param rowIndex2 index of second row to swap.
     *
     * @return A reference to this matrix.
     */
    @Override
    public CsrCMatrix swapRows(int rowIndex1, int rowIndex2) {
        ComplexCsrManipulations.swapRows(this, rowIndex1, rowIndex2);
        return this;
    }


    /**
     * Swaps columns in the matrix.
     *
     * @param colIndex1 Index of first column to swap.
     * @param colIndex2 index of second column to swap.
     *
     * @return A reference to this matrix.
     */
    @Override
    public CsrCMatrix swapCols(int colIndex1, int colIndex2) {
        ComplexCsrManipulations.swapCols(this, colIndex1, colIndex2);
        return this;
    }


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    @Override
    public boolean isSquare() {
        return numRows == numCols;
    }


    /**
     * Checks if this CSR matrix is equal to another CSR matrix.
     * @param src2 Object to compare this matrix to.
     * @return True if {@code src2} is an instance of {@link CsrMatrix} this CSR matrix is equal to {@code src2}.
     * False otherwise. If {@code src2} is null, false is returned.
     */
    public boolean equals(Object src2) {
        // Check for early returns.
        if(this == src2) return true;
        if(src2 == null || src2.getClass() != getClass()) return false;

        CsrCMatrix b = (CsrCMatrix) src2;

        if(entries.length == b.entries.length) {
            // Arrays can be directly compared (even with explicitly stored zeros).
            return shape.equals(b.shape)
                    && Arrays.equals(entries, b.entries)
                    && Arrays.equals(rowPointers, b.rowPointers)
                    && Arrays.equals(colIndices, b.colIndices);
        } else {
            // Then possible explicitly stored zero value must be considered
            // (e.g. one matrix explicitly stores a zero at some position and the other does not).
            return SparseUtils.CSREquals(this, b);
        }
    }


    @Override
    public int hashCode() {
        int result = 17;

        result = 31 * result + numRows;
        result = 31 * result + numCols;
        result = 31 * result + Arrays.hashCode(rowPointers);
        result = 31 * result + Arrays.hashCode(colIndices);
        result = 31 * result + Arrays.hashCode(entries);
        return result;
    }


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     *
     * @return True if this matrix can be represented as either a row or column vector.
     */
    @Override
    public boolean isVector() {
        return numRows==1 || numCols==1;
    }


    /**
     * Checks what type of vector this matrix is. i.e. not a vector, a 1x1 matrix, a row vector, or a column vector.
     *
     * @return - If this matrix can not be represented as a vector, then returns -1. <br>
     * - If this matrix is a 1x1 matrix, then returns 0. <br>
     * - If this matrix is a row vector, then returns 1. <br>
     * - If this matrix is a column vector, then returns 2.
     */
    @Override
    public int vectorType() {
        int type = -1;

        if(numRows==1 && numCols==1) type=0;
        else if(numRows==1) type=1;
        else if(numCols==1) type=2;

        return type;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriL() {
        boolean result = isSquare();
        CNumber zero = CNumber.zero();

        if(result) {
            for(int i=0; i<numRows; i++) {
                int rowStart = rowPointers[i];
                int rowStop = rowPointers[i+1];

                for(int j=rowStop-1; j>=rowStart; j--) {
                    if(colIndices[j] <= i) {
                        break; // Have reached the diagonal. No need to continue for this row.
                    } else if(!entries[j].equals(zero)) {
                        return false; // Non-zero entry found. No need to continue.
                    }
                }
            }
        }

        return result;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriU() {
        boolean result = isSquare();
        CNumber zero = CNumber.zero();

        if(result) {
            for(int i=1; i<numRows; i++) {
                int rowStart = rowPointers[i];
                int stop = rowPointers[i+1];

                for(int j=rowStart; j<stop; j++) {
                    if(colIndices[j] >= i) {
                        break; // Have reached the diagonal. No need to continue for this row.
                    } else if(!entries[j].equals(zero)) {
                        return false; // Non-zero entry found. No need to continue.
                    }
                }
            }
        }

        return result;
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return matrixRank() == Math.min(numRows, numCols);
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     *
     * @return True if this matrix is singular. Otherwise, returns false.
     * @see #isInvertible()
     */
    @Override
    public boolean isSingular() {
        return toDense().isSingular();
    }


    /**
     * <p>Computes the rank of this matrix (i.e. the dimension of the column space of this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank.</p>
     *
     * <p><b>WARNING</b>: This method will convert this matrix to a dense matrix to compute the rank.</p>
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        // TODO: Investigate methods of rank computation for sparse matrices.
        return toDense().matrixRank();
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nonZeroEntries;
        StringBuilder result = new StringBuilder(String.format("Full Shape: %s\n", shape));
        result.append("Non-zero entries: [");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        if(entries.length > 0) {
            // Get entries up until the stopping point.
            for(int i=0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
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
            value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
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
