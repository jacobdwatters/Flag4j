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

package org.flag4j.sparse;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.MatrixMixin;
import org.flag4j.core.RealMatrixMixin;
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.core.sparse_base.RealSparseTensorBase;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Matrix;
import org.flag4j.dense.Vector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.decompositions.svd.SVD;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.dense.real.RealDenseOperations;
import org.flag4j.operations.dense_sparse.csr.real.RealCsrDenseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.real.RealCsrDenseOperations;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseOperations;
import org.flag4j.operations.sparse.csr.real.RealCsrEquals;
import org.flag4j.operations.sparse.csr.real.RealCsrMatrixMultiplication;
import org.flag4j.operations.sparse.csr.real.RealCsrMatrixProperties;
import org.flag4j.operations.sparse.csr.real.RealCsrOperations;
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
 * Real sparse matrix stored in compressed sparse row (CSR) format.<br><br>
 *
 * CSR matrices are best suited for efficient access and matrix operations. Specifically, matrix-matrix and
 * matrix-vector multiplication. CSR matrices are <b>not</b> well suited for modification (see {@link CooMatrix}).
 * <br><br>
 *
 * The CSR format stores a sparse <code>m-by-n</code> matrix as three one-dimensional arrays: {@link #entries},
 * {@link #rowPointers}, and {@link #colIndices}.<br>
 * <pre>
 *   <b>- entries</b>: Stores the non-zero values of the sparse matrix. Note, zero values can be
 *   stored explicitly in this array. Hence, the term "non-zero values" is a misnomer.
 *   <b>- rowPointers</b>: Encodes the total number of non-zero values above each row.
 *     Has length <code>m+1</code>. For example, <code>rowPointers[j]</code> contains
 *     the total number of non-zero values above row <code>j</code>.
 *     The first entry is always 0 and the last element is always <code>entries.length</code>
 *   <b>- colIndices</b>: Contains the column indices for all non-zero entries. Has length <code>entries.length</code>
 * </pre>
 * @see CooMatrix
 */
public class CsrMatrix
        extends RealSparseTensorBase<CsrMatrix, Matrix, CsrCMatrix, CMatrix>
        implements MatrixMixin<CsrMatrix, Matrix, CsrMatrix, CsrCMatrix, Double, CooVector, Vector>,
        RealMatrixMixin<CsrMatrix, CsrCMatrix>
{


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
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrMatrix(Shape shape, double[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries.length, entries, new int[colIndices.length], colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
        nnz = entries.length;
    }


    /**
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param numRows Number of rows in the CSR matrix.
     * @param numCols Number of columns in the CSR matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrMatrix(int numRows, int numCols, double[] entries, int[] rowPointers, int[] colIndices) {
        super(new Shape(numRows, numCols), entries.length, entries, rowPointers, colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        nnz = entries.length;
    }


    public CsrMatrix(CsrMatrix src) {
        super(src.shape.copy(), src.entries.length, src.entries.clone(),
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
    public CsrMatrix(CooMatrix src) {
        super(src.shape.copy(),
                src.entries.length,
                src.entries.clone(),
                new int[src.numRows + 1],
                src.colIndices.clone()
        );

        rowPointers = this.indices[0];
        colIndices = this.indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
        nnz = entries.length;

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
     * Checks if this CSR matrix is equal to another CSR matrix.
     * @param src2 Object to compare this matrix to.
     * @return True if {@code src2} is an instance of {@link CsrMatrix} this CSR matrix is equal to {@code src2}.
     * False otherwise. If {@code src2} is null, false is returned.
     */
    public boolean equals(Object src2) {
        // TODO: The equals method in the other matrix, vector, and tensor classes should only accept their
        //  own type as well. Add an additional static equals method like Arrays.equals(...)
        //  to check numerical equality. The Object.equals method should also check type.
        if(this == src2) return true;
        if(src2 == null) return false;
        if(!(src2 instanceof CsrMatrix)) return false;

        CsrMatrix b = (CsrMatrix) src2;

        // TODO: If one matrix stores explicit zeros and the other does not, this may fail.
        //  Add a dropZeros() method to remove any explicitly stored zeros.
        return shape.equals(b.shape)
                && Arrays.equals(entries, b.entries)
                && Arrays.equals(rowPointers, b.rowPointers)
                && Arrays.equals(colIndices, b.colIndices);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CsrMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrMatrix mult2CSR(CsrMatrix B) {
        return RealCsrMatrixMultiplication.standardAsSparse(this, B);
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
        return RealComplexCsrMatrixMultiplication.standardAsSparse(this, B);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and stores the result in a CSR matrix.
     * Warning: This method will likely be slower than {@link #mult(CooMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrMatrix mult2CSR(CooMatrix B) {
        return RealCsrMatrixMultiplication.standardAsSparse(this, B.toCsr());
    }


    /**
     * Computes the element-wise addition between two matrices.
     *
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix add(Matrix B) {
        return RealCsrDenseOperations.applyBinOpp(this, B, Double::sum);
    }


    /**
     * Computes the element-wise addition between two sparse matrices. Note that the CooMatrix is simply converted to
     * a csr matrix first.
     *
     * @param B Second sparse matrix in the sum.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     * @implNote Implemented as <code>this.add(B.toCsr())</code>
     */
    @Override
    public CsrMatrix add(CooMatrix B) {
        return this.add(B.toCsr());
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(CMatrix B) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, B, (Double a, CNumber b) -> b.add(a));
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CsrCMatrix add(CooCMatrix B) {
        return this.add(B.toCsr());
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public CsrCMatrix add(CsrCMatrix B) {
        return RealComplexCsrOperations.applyBinOpp(this, B, (Double a, CNumber b)->b.add(a));
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix sub(Matrix B) {
        return RealCsrDenseOperations.applyBinOpp(this, B,(Double a, Double b)->a-b);
    }


    /**
     * Computes the element-wise subtraction of two sparse matrices. Note, the CooMatrix is simply converted
     * to a CsrMatrix first.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     * @implNote <code>this.sub(B.toCsr())</code>
     */
    @Override
    public CsrMatrix sub(CooMatrix B) {
        return this.sub(B.toCsr());
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(CMatrix B) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, B, (Double a, CNumber b)->new CNumber(a).sub(b));
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CsrCMatrix sub(CooCMatrix B) {
        return this.sub(B.toCsr());
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public CsrCMatrix sub(CsrCMatrix B) {
        return RealComplexCsrOperations.applyBinOpp(this, B, (Double a, CNumber b)->new CNumber(a).sub(b));
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  rows in matrix {@code B}.
     */
    @Override
    public Matrix mult(Matrix B) {
        return RealCsrDenseMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix-vector multiplication.
     *
     * @param B Vector to multiply this matrix to.
     * @return The vector result from multiplying this matrix by the vector {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     *                                  entries {@code B}.
     */
    @Override
    public Vector mult(CooVector B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(CVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CVector mult(CooCVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(Matrix B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public Matrix multTranspose(CooMatrix B) {
        return this.multTranspose(B.toCsr());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    public Matrix multTranspose(CsrMatrix B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {{@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CMatrix B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    @Override
    public CMatrix multTranspose(CooCMatrix B) {
        return this.multTranspose(B.toCsr());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     *
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    public CMatrix multTranspose(CsrCMatrix B) {
        return this.mult(B.T());
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times.
     *
     * @param exponent The exponent in the matrix power.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     */
    @Override
    public Matrix pow(int exponent) {
        ParameterChecks.assertPositive(exponent);
        // TODO: Implementation.

        if(exponent==0) {
            return new Matrix(this.shape.copy());
        } else if(exponent==1) {
            return this.toDense();
        }
        else {
            Matrix exp = this.mult(this); // First multiplication is sparse-sparse multiplication.

            for(int i=2; i<exponent; i++) {
//                exp.mult(this); // The Matrix.Mult(CsrMatrix) method needs to be implemented in the Matrix class.
            }

            return exp;
        }
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CsrMatrix elemMult(Matrix B) {
        return RealCsrDenseOperations.applyBinOppToSparse(B, this, (Double a, Double b)->a*b);
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CsrMatrix elemMult(CooMatrix B) {
        return this.elemMult(B.toCsr());
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CsrCMatrix elemMult(CMatrix B) {
        return RealComplexCsrDenseOperations.applyBinOppToSparse(B, this, CNumber::mult);
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CooCMatrix elemMult(CooCMatrix B) {
        return this.elemMult(B.toCsr()).toCoo();
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public CsrCMatrix elemMult(CsrCMatrix B) {
        return RealComplexCsrOperations.elemMult(B, this);
    }


    /**
     * Computes the element-wise division between two matrices.
     *
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException      If B contains any zero entries.
     */
    @Override
    public CsrCMatrix elemDiv(CMatrix B) {
        return RealComplexCsrDenseOperations.applyBinOppToSparse(this, B,
                (Double a, CNumber b) -> new CNumber(a).div(b));
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * <p><b>WARNING:</b> Currently, this method will convert this matrix to a dense matrix.</p>
     *
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double det() {
        return toDense().det();
    }


    /**
     * Constructs an identity-like matrix stored in CSR format with the specified shape.
     * @param shape Shape of the identity-like matrix.
     * @return An identity-like matrix stored in CSR format.
     */
    public static CsrMatrix I(Shape shape) {
        return I(shape.dims[0], shape.dims[1]);
    }


    /**
     * Constructs an identity matrix stored in CSR format with the specified shape.
     * @param size Number of rows (and columns) in the matrix.
     * @return An identity matrix stored in CSR format.
     */
    public static CsrMatrix I(int size) {
        return I(size, size);
    }


    /**
     * Constructs an identity-like matrix stored in CSR format with the specified shape.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return An identity-like matrix stored in CSR format.
     */
    public static CsrMatrix I(int numRows, int numCols) {
        int nnz = Math.min(numRows, numCols);
        double[] entries = new double[nnz];
        Arrays.fill(entries, 1);
        int[] rowPointers = new int[numRows + 1];
        int[] colIndices = ArrayUtils.intRange(0, nnz);

        for(int i=1; i<=nnz; i++) {
            rowPointers[i] = i;
        }

        Arrays.fill(rowPointers, nnz+1, rowPointers.length, nnz);

        return new CsrMatrix(numRows, numCols, entries, rowPointers, colIndices);
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(Matrix B) {
        return this.T().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(CooMatrix B) {
        return this.fib(B.toCsr());
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public Double fib(CsrMatrix B) {
        return this.T().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public CNumber fib(CsrCMatrix B) {
        return this.T().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CMatrix B) {
        return this.T().mult(B).tr();
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CooCMatrix B) {
        return fib(B.toCsr());
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CsrMatrix directSum(Matrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CsrMatrix directSum(CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CsrCMatrix directSum(CMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CsrCMatrix directSum(CooCMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CsrMatrix invDirectSum(Matrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CsrMatrix invDirectSum(CsrMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CsrMatrix invDirectSum(CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CsrCMatrix invDirectSum(CMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CsrCMatrix invDirectSum(CooCMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    @Override
    public CsrMatrix sumCols() {
        // TODO: Implementation
        return null;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    @Override
    public CsrMatrix sumRows() {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(Vector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(CooVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CooCVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(Vector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(CooVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CooCVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public Matrix stack(Matrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CsrMatrix stack(CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(CMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CsrCMatrix stack(CooCMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public Matrix augment(Matrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CsrMatrix augment(CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(CMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CsrCMatrix augment(CooCMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(Vector, int)} and {@link #augment(Vector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrMatrix stack(Vector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooVector, int)} and {@link #augment(CooVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrMatrix stack(CooVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CVector, int)} and {@link #augment(CVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrCMatrix stack(CVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooCVector, int)} and {@link #augment(CooCVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CsrCMatrix stack(CooCVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(Vector)} and {@link #stack(Vector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrMatrix augment(Vector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooVector)} and {@link #stack(CooVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrMatrix augment(CooVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CVector)} and {@link #stack(CVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrCMatrix augment(CVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooCVector)} and {@link #stack(CooCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CsrCMatrix augment(CooCVector b) {
        // TODO: Implementation
        return null;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    @Override
    public CooVector getRow(int i) {
        int start = rowPointers[i];

        double[] destEntries = new double[rowPointers[i+1]-start];
        int[] destIndices = new int[destEntries.length];

        System.arraycopy(entries, start, destEntries, 0, destEntries.length);
        System.arraycopy(colIndices, start, destIndices, 0, destEntries.length);

        return new CooVector(this.numCols, destEntries, destIndices);
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public CooVector getCol(int j) {
        return getCol(j, 0, numRows);
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param j   Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd   Ending row of the column (exclusive).
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     * @throws IllegalArgumentException   If {@code rowStart} is less than 0.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public CooVector getCol(int j, int rowStart, int rowEnd) {
        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=rowStart; i<rowEnd; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int colIdx=start; colIdx<stop; colIdx++) {
                if(colIndices[colIdx]==j) {
                    destEntries.add(entries[colIdx]);
                    destIndices.add(i);
                    break; // Should only be a single entry with this row and column index.
                }
            }
        }

        return new CooVector(numRows, destEntries, destIndices);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooVector toVector() {
        int type = vectorType();

        double[] destEntries = this.entries.clone(); // Copy non-zero values.
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

        return new CooVector(shape.totalEntries().intValueExact(), destEntries, indices);
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd   Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd   Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public CsrMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        // TODO: Implementation
        return null;
    }


    /**
     * Get a specified column of this matrix at and below a specified row.
     *
     * @param rowStart Index of the row to begin at.
     * @param j        Index of column to get.
     * @return The specified column of this matrix beginning at the specified row.
     * @throws NegativeArraySizeException     If {@code rowStart} is larger than the number of rows in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowStart} or {@code j} is outside the bounds of this matrix.
     */
    @Override
    public CooVector getColBelow(int rowStart, int j) {
        return getCol(j, rowStart, numRows);
    }


    /**
     * Get a specified row of this matrix at and after a specified column.
     *
     * @param colStart Index of the row to begin at.
     * @param i        Index of the row to get.
     * @return The specified row of this matrix beginning at the specified column.
     * @throws NegativeArraySizeException     If {@code colStart} is larger than the number of columns in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code i} or {@code colStart} is outside the bounds of this matrix.
     */
    @Override
    public CooVector getRowAfter(int colStart, int i) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a column of this matrix.
     *
     * @param values Vector containing the new values for the matrix.
     * @param j      Index of the column of this matrix to set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException  If the number of entries in the {@code values} vector
     *                                   is not the same as the number of rows in this matrix.
     * @throws IndexOutOfBoundsException If {@code j} is not within the bounds of this matrix.
     */
    @Override
    public CsrMatrix setCol(CooVector values, int j) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #tr()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double trace() {
        return tr();
    }


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double tr() {
        ParameterChecks.assertSquareMatrix(shape);

        double trace = 0;

        for(int i=0; i<numRows; i++) {
            int rowPtr = rowPointers[i];
            int stop = rowPointers[i+1];

            for(int j=rowPtr; j<stop; j++) {
                if(i==colIndices[j]) {
                    trace++;
                }
            }
        }

        return trace;
    }


    /**
     * Computes the inverse of this matrix. Warning: Currently, this method converts the matrix to a dense matrix
     * first then computes the inverse using a standard dense algorithm {@link Matrix#inv()}.
     *
     * @return The inverse of this matrix.
     */
    @Override
    public Matrix inv() {
        // TODO: Should have specialized algorithms for inverting sparse matrix.
        return toDense().inv();
    }


    /**
     * Computes the pseudo-inverse of this matrix. Warning: Currently, this method converts the matrix to a dense matrix
     * first then computes the pseudo-inverse using a standard dense algorithm {@link Matrix#pInv()}.
     *
     * @return The pseudo-inverse of this matrix.
     */
    @Override
    public Matrix pInv() {
        // TODO: Should have specialized algorithms for pseudo-inverting sparse matrix.
        return toDense().pInv();
    }


    /**
     * Computes the condition number of this matrix using {@link SVD SVD}.
     * Specifically, the condition number is computed as the maximum singular value divided by the minimum singular
     * value of this matrix.
     *
     * @return The condition number of this matrix (Assuming Frobenius norm).
     */
    @Override
    public double cond() {
        return toDense().cond();
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public CooVector getDiag() {
        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0; i<numRows; i++) {
            int start = rowPointers[i];
            int stop = rowPointers[i+1];

            int loc = Arrays.binarySearch(colIndices, i, start, stop); // Search for matching column index

            if(loc > 0) {
                destEntries.add(entries[loc]);
                destIndices.add(i);
            }
        }

        return new CooVector(Math.min(numRows, numCols), destEntries, destIndices);
    }


    /**
     * Compute the hermation transpose of this matrix. That is, the complex conjugate transpose of this matrix.
     *
     * @return The complex conjugate transpose of this matrix.
     */
    @Override
    public CsrMatrix H() {
        return T();
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is stored as a dense matrix. <br><br>
     *
     * If memory is a concern, and you are confident the multiplication result will be sparse, consider using {@link #mult2CSR(CsrMatrix)}
     * which will store the result as a {@link CsrMatrix}. However, the method should be used with caution as it will
     * almost never be faster than this method and sparse matrix multiplication may result in fully dense matrices
     * (even for two very sparse matrices).
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * rows in matrix {@code B}.
     */
    public Matrix mult(CsrMatrix B) {
        return RealCsrMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is stored as a dense matrix. <br><br>
     *
     * If memory is a concern, and you are confident the multiplication result will be sparse, consider using {@link #mult2CSR(CsrCMatrix)}
     * which will store the result as a {@link CsrCMatrix}. However, the method should be used with caution as it will
     * almost never be faster than this method and sparse matrix multiplication may result in fully dense matrices
     * (even for two very sparse matrices).
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * rows in matrix {@code B}.
     */
    public CMatrix mult(CsrCMatrix B) {
        return RealComplexCsrMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(CooMatrix B) {
        return this.mult(B.toCsr());
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CMatrix B) {
        return RealComplexCsrDenseMatrixMultiplication.standard(this, B);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CooCMatrix B) {
        return mult(B.toCsr());
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Vector mult(Vector b) {
        return RealCsrDenseMatrixMultiplication.standardVector(this, b);
    }


    /**
     * Checks if this matrix is the identity matrix.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    public boolean isI() {
        return RealCsrMatrixProperties.isIdentity(this);
    }

    
    /**
     * Checks if matrices are inverses of each other.
     *
     * @param B Second matrix.
     * @return True if matrix B is an inverse of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    @Override
    public boolean isInv(CsrMatrix B) {
        boolean result;

        if(!this.isSquare() || !B.isSquare() || !shape.equals(B.shape)) {
            result = false;
        } else {
            result = this.mult(B).isCloseToI();
        }

        return result;
    }


    /**
     * Sets an index of this matrix to the specified value. Note: new entries cannot be inserted into a CSR matrix.
     * This method returns a new matrix. In general, calling this method repeatedly is <b>not</b> considered
     * to be efficient.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A new CSR matrix with the specified value set.
     */
    @Override
    public CsrMatrix set(double value, int row, int col) {
        // Ensure indices are in bounds.
        ParameterChecks.assertValidIndex(shape, row, col);
        double[] newEntries;
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
            newEntries = entries.clone();
            newEntries[loc] = value;
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch
            newEntries = new double[entries.length + 1];
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

        return new CsrMatrix(shape.copy(), newEntries, newRowPointers, newColIndices);
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row   Row index to set.
     * @param col   Column index to set.
     * @return A reference to this matrix.
     */
    @Override
    public CsrMatrix set(Double value, int row, int col) {
        return set(value.doubleValue(), row, col);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrMatrix setCol(Double[] values, int colIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrMatrix setCol(Integer[] values, int colIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrMatrix setCol(double[] values, int colIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public CsrMatrix setCol(int[] values, int colIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrMatrix setRow(Double[] values, int rowIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrMatrix setRow(Integer[] values, int rowIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrMatrix setRow(double[] values, int rowIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @return A reference to this matrix.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public CsrMatrix setRow(int[] values, int rowIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(CsrMatrix values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(Matrix values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(CooMatrix values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(Double[][] values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(Integer[][] values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(double[][] values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A reference to this matrix.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public CsrMatrix setSlice(int[][] values, int rowStart, int colStart) {
        // TODO: Implementation
        return null;
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     * @return a copy of this matrix with the specified row removed.
     */
    @Override
    public CsrMatrix removeRow(int rowIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix.
     * @return a copy of this matrix with the specified rows removed.
     */
    @Override
    public CsrMatrix removeRows(int... rowIndices) {
        // TODO: Implementation
        return null;
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CsrMatrix removeCol(int colIndex) {
        // TODO: Implementation
        return null;
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     * @return a copy of this matrix with the specified columns removed.
     */
    @Override
    public CsrMatrix removeCols(int... colIndices) {
        // TODO: Implementation
        return null;
    }


    /**
     * Swaps rows in the matrix.
     *
     * @param rowIndex1 Index of first row to swap.
     * @param rowIndex2 index of second row to swap.
     * @return A reference to this matrix.
     */
    @Override
    public CsrMatrix swapRows(int rowIndex1, int rowIndex2) {
        // TODO: Implementation
        return null;
    }


    /**
     * Swaps columns in the matrix.
     *
     * @param colIndex1 Index of first column to swap.
     * @param colIndex2 index of second column to swap.
     * @return A reference to this matrix.
     */
    @Override
    public CsrMatrix swapCols(int colIndex1, int colIndex2) {
        // TODO: Implementation
        return null;
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
     * Gets the shape of this matrix.
     *
     * @return The shape of this matrix.
     */
    @Override
    public Shape shape() {
        return shape;
    }


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    @Override
    public boolean isSquare() {
        return numRows==numCols;
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

        if(numRows==1 && numCols==1) {
            type=0;
        } else if(numRows==1) {
            type=1;
        } else if(numCols==1) {
            type=2;
        }

        return type;
    }


    /**
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     *
     * @return True is this matrix is triangular. Otherwise, returns false.
     * @see #isTriL()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTri() {
        return isTriL() || isTriU();
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        boolean result = isSquare();

        if(result) {
            for(int i=0; i<numRows; i++) {
                int rowStart = rowPointers[i];
                int rowStop = rowPointers[i+1];

                for(int j=rowStop-1; j>=rowStart; j--) {
                    if(colIndices[j] <= i) {
                        break; // Have reached the diagonal. No need to continue for this row.
                    } else if(entries[j] != 0) {
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
     * @see #isTriL()
     * @see #isTri()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        boolean result = isSquare();

        if(result) {
            for(int i=1; i<numRows; i++) {
                int rowStart = rowPointers[i];
                int stop = rowPointers[i+1];

                for(int j=rowStart; j<stop; j++) {
                    if(colIndices[j] >= i) {
                        break; // Have reached the diagonal. No need to continue for this row.
                    } else if(entries[j] != 0) {
                        return false; // Non-zero entry found. No need to continue.
                    }
                }
            }
        }

        return result;
    }


    /**
     * Checks if this matrix is diagonal.
     *
     * @return True is this matrix is diagonal. Otherwise, returns false.
     * @see #isTriL()
     * @see #isTriU()
     * @see #isTri()
     */
    @Override
    public boolean isDiag() {
        return isTriL() && isTriU();
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return toDense().isFullRank();
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     *
     * @return True if this matrix is singular. Otherwise, returns false.
     */
    @Override
    public boolean isSingular() {
        return toDense().isSingular();
    }


    /**
     * Checks if a matrix is invertible.<br>
     * Also see {@link #isSingular()}.
     *
     * @return True if this matrix is invertible.
     */
    @Override
    public boolean isInvertible() {
        return !isSingular();
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    @Override
    public double norm(double p, double q) {
        return RealCsrOperations.matrixNormLpq(this, p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @return The max norm of this matrix.
     */
    @Override
    public double maxNorm() {
        return RealDenseOperations.matrixMaxNorm(entries);
    }


    /**
     * Computes the rank of this matrix (i.e. the dimension of the column space of this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank.
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        return toDense().matrixRank();
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        // TODO: Implementation
        return false;
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isAntiSymmetric() {
        // TODO: Implementation
        return false;
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        return isSquare() && this.mult(this.T()).allClose(Matrix.I(numRows));
    }


    /**
     * Computes the complex element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public CsrCMatrix sqrtComplex() {
        return new CsrCMatrix(shape.copy(), ComplexOperations.sqrt(entries), rowPointers.clone(), colIndices.clone());
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CsrCMatrix toComplex() {
        return new CsrCMatrix(shape.copy(), ComplexOperations.sqrt(entries), rowPointers.clone(), colIndices.clone());
    }


    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CsrMatrix getSelf() {
        return this;
    }


    /**
     * Converts this {@link CsrMatrix CSR matrix} to an equivalent {@link CooMatrix COO matrix}.
     * @return A {@link CooMatrix COO matrix} equivalent to this {@link CsrMatrix CSR matrix}.
     */
    public CooMatrix toCoo() {
        double[] dest = entries.clone();
        int[] destRowIdx = new int[entries.length];
        int[] destColIdx = colIndices.clone();

        for(int i=0; i<numRows; i++) {
            int stop = rowPointers[i+1];

            for(int j=rowPointers[i]; j<stop; j++) {
                destRowIdx[j] = i;
            }
        }

        return new CooMatrix(shape.copy(), dest, destRowIdx, destColIdx);
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     *
     * @param src Tensor to compare this tensor to.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (atol + rtol*|b|)}. Otherwise, returns false.
     * @see TensorBase#allClose(Object, double, double)
     */
    @Override
    public boolean allClose(CsrMatrix src, double relTol, double absTol) {
        return RealCsrEquals.allClose(this, src, relTol, absTol);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public CsrMatrix set(double value, int... indices) {
        ParameterChecks.assertValidIndex(shape, indices);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public CsrMatrix flatten(int axis) {
        if(axis==0) {
            // Flatten to single row
            return reshape(new Shape(1, entries.length));
        } else if(axis==1) {
            // Flatten to single column
            return reshape(new Shape(entries.length, 1));
        } else {
            // Unknown axis
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
    public CsrMatrix add(CsrMatrix B) {
        return RealCsrOperations.applyBinOpp(this, B, Double::sum);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix add(double a) {
        return RealCsrDenseOperations.applyBinOpp(this, a, Double::sum);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, a, (Double x, CNumber y)->y.add(x));
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CsrMatrix sub(CsrMatrix B) {
        return RealCsrOperations.applyBinOpp(this, B, (Double a, Double b) -> a-b);
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix sub(double a) {
        return RealCsrDenseOperations.applyBinOpp(this, a, (Double x, Double y)->x-y);
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, a, (Double x, CNumber y)->new CNumber(x).sub(y));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CsrMatrix T() {
        return RealCsrOperations.transpose(this);
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public Double get(int... indices) {
        ParameterChecks.assertValidIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];
        int loc = Arrays.binarySearch(colIndices, rowPointers[row], rowPointers[row+1], col);

        if(loc > 0) return entries[loc];
        else return 0d;
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CsrMatrix copy() {
        return new CsrMatrix(this);
    }


    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CsrMatrix elemMult(CsrMatrix B) {
        return RealCsrOperations.applyBinOpp(this, B, (Double a, Double b) -> a*b);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CsrMatrix elemDiv(Matrix B) {
        return RealCsrDenseOperations.applyBinOppToSparse(B, this, (Double x, Double y)->y/x);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return RealDenseOperations.tensorNormL2(entries); // Zeros do not contribute to this norm.
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    @Override
    public double norm(double p) {
        return RealDenseOperations.tensorNormLp(entries, p); // Zeros do not contribute to this norm.
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
    protected CsrMatrix makeTensor(Shape shape, double[] entries, int[][] indices) {
        return new CsrMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * A factory for creating a real dense tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Entries of the dense tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected Matrix makeDenseTensor(Shape shape, double[] entries) {
        return new Matrix(shape, entries);
    }


    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CsrCMatrix makeComplexTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return new CsrCMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
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

        return new Matrix(shape.copy(), dest);
    }


    /**
     * Formats this sparse CSR matrix as a human-readable string.
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
