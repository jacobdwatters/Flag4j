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

package org.flag4j.arrays.dense;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.sparse.*;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.MatrixMultiplyDispatcher;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatMult;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatMult;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.csr.field_ops.DenseCsrFieldMatMult;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrMatMult;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>Instances of this class represents a complex dense matrix backed by a {@link Complex128} array. The {@code CMatrix} class
 * provides functionality for complex matrix operations, supporting mutable data with a fixed shape.
 * This class extends {@link AbstractDenseFieldMatrix} and offers additional methods optimized for complex
 * arithmetic and matrix computations.
 *
 * <p>A {@code CMatrix} is essentially equivalent to a rank-2 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general tensors.
 *
 * <p><b>Key Features:</b>
 * <ul>
 *   <li>Construction from various data types such as arrays of {@link Complex128}, {@code double}, and {@link String}.</li>
 *   <li>Support for standard matrix operations like addition, subtraction, multiplication, and exponentiation.</li>
 *   <li>Conversion methods to other matrix representations, such as COO (Coordinate) and CSR (Compressed Sparse Row) formats.</li>
 *   <li>Utility methods for checking properties like being unitary, real, or complex.</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Constructing a complex matrix from a 2D array of complex numbers
 * Complex128[][] complexData = {
 *     { new Complex128(1, 2), new Complex128(3, 4) },
 *     { new Complex128(5, 6), new Complex128(7, 8) }
 * };
 * CMatrix matrix = new CMatrix(complexData);
 *
 * // Performing matrix multiplication.
 * CMatrix result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * CMatrix transpose = matrix.T();
 *
 * // Performing matrix conjugate transpose (i.e. Hermitian transpose).
 * CMatrix conjugateTranspose = matrix.H();
 *
 * // Checking if the matrix is unitary.
 * boolean isUnitary = matrix.isUnitary();
 * }</pre>
 *
 * @see Complex128
 * @see CVector
 * @see AbstractDenseFieldMatrix
 */
public class CMatrix extends AbstractDenseFieldMatrix<CMatrix, CVector, Complex128> {
    private static final long serialVersionUID = 1L;


    /**
     * Creates a complex matrix with the specified {@code data} and {@code shape}.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(Shape shape, Complex128[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code shape} filled with {@code fillValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Value to fill this matrix with.
     */
    public CMatrix(Shape shape, Complex128 fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a zero matrix with the specified {@code shape}.
     *
     * @param shape Shape of this matrix.
     */
    public CMatrix(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Creates a square zero matrix with the specified {@code size}.
     *
     * @param size Size of the zero matrix to construct. The resulting matrix will have shape {@code (size, size)}
     */
    public CMatrix(int size) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code data}, and shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(int rows, int cols, Complex128[] entries) {
        super(new Shape(rows, cols), entries);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified shape and filled with {@code fillValue}.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param fillValue Value to fill this matrix with.
     */
    public CMatrix(int rows, int cols, Complex128 fillValue) {
        super(new Shape(rows, cols), new Complex128[rows*cols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a zero matrix with the specified shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     */
    public CMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[rows*cols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Constructs a complex matrix from a 2D array. The matrix will have the same shape as the array.
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(Complex128[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(Complex128[] row : entries) {
            for(Complex128 value : row)
                super.data[flatPos++] = value;
        }
    }


    /**
     * <p>Constructs a complex matrix from a 2D array of strings. Each string must be formatted properly as a complex number that can
     * be parsed by {@link org.flag4j.io.parsing.ComplexNumberParser}
     *
     * <p>The matrix will have the same shape as the array.
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(String[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(String[] row : entries) {
            for(String value : row)
                super.data[flatPos++] = new Complex128(value);
        }
    }


    /**
     * Constructs a complex matrix with specified {@code shape} and {@code data}.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix.
     */
    public CMatrix(Shape shape, double[] entries) {
        super(shape, new Complex128[entries.length]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
        ArrayUtils.arraycopy(entries, 0, super.data, 0, entries.length);
    }


    /**
     * Constructs a complex matrix from a 2D array of double values.
     * @param aEntriesReal Entries of the complex matrix to construct. Each value will be wrapped as {@link Complex128 Complex128's}.
     */
    public CMatrix(double[][] aEntriesReal) {
        super(new Shape(aEntriesReal.length, aEntriesReal[0].length), new Complex128[aEntriesReal.length*aEntriesReal[0].length]);
        setZeroElement(Complex128.ZERO);

        int idx = 0;
        for(double[] row : aEntriesReal) {
            for(double value : row)
                super.data[idx++] = new Complex128(value);
        }
    }


    /**
     * Constructs a matrix with the specified shape filled with {@code fillValue}.
     * @param numRows The number of rows in the matrix.
     * @param numCols The number of rows in the matrix.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int numRows, int numCols, double fillValue) {
        super(new Shape(numRows, numCols), new Complex128[numRows*numCols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Complex128 fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Double fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates matrix with the specified {@code shape} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(Shape shape, Double fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Constructs a copy of the specified matrix.
     * @param mat Matrix to create copy of.
     */
    public CMatrix(CMatrix mat) {
        super(mat.shape, mat.data.clone());
    }


    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of a similar type as this matrix.
     */
    @Override
    protected CVector makeLikeVector(Complex128[] entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a sparse COO matrix which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     *
     * @return A sparse COO matrix which is of a similar type as this dense matrix.
     */
    @Override
    protected CooCMatrix makeLikeCooMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a sparse CSR matrix which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A sparse CSR matrix which is of a similar type as this dense matrix.
     */
    @Override
    public CsrCMatrix makeLikeCsrMatrix(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCMatrix toCoo() {
        return (CooCMatrix) super.toCoo();
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     *
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     *
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     *
     * @see #toCoo()
     */
    @Override
    public CooCMatrix toCoo(double estimatedSparsity) {
        return (CooCMatrix) super.toCoo(estimatedSparsity);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr(double)
     */
    @Override
    public CsrCMatrix toCsr() {
        return (CsrCMatrix) super.toCsr();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr()
     */
    @Override
    public CsrCMatrix toCsr(double estimatedSparsity) {
        return (CsrCMatrix) super.toCsr(estimatedSparsity);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooCMatrix makeLikeCooTensor(Shape shape, Complex128[] entries, int[][] indices) {
        return makeLikeCooMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CMatrix makeLikeTensor(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs an empty matrix with the specified number of rows and columns. The data of the resulting matrix will be
     * all be {@code null}.
     * @param rows The number of rows in the matrix to construct.
     * @param cols The number of columns in the matrix to construct.
     * @return An empty matrix (i.e. filled with {@code null} values) with the specified shape.
     */
    public static CMatrix getEmpty(int rows, int cols) {
        return new CMatrix(rows, cols, new Complex128[rows*cols]);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix add(CooCMatrix b) {
        return (CMatrix) DenseCooFieldMatrixOps.add(this, b);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix add(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix add(CooMatrix b) {
        return (CMatrix) RealFieldDenseCooMatrixOps.add(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix sub(CooCMatrix b) {
        return (CMatrix) DenseCooFieldMatrixOps.sub(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix sub(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }



    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix sub(CooMatrix b) {
        return (CMatrix) RealFieldDenseCooMatrixOps.sub(this, b);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(CooCMatrix b) {
        Complex128[] dest = new Complex128[b.nnz];
        DenseCooFieldMatrixOps.elemMult(shape, data, b.shape, b.data, b.rowIndices, b.colIndices, dest);
        return new CooCMatrix(shape, dest, b.rowIndices.clone(), b.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix elemMult(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemMult(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(CooMatrix b) {
        Complex128[] dest = new Complex128[b.nnz];
        RealFieldDenseCooMatrixOps.elemMult(this, b, dest);
        return new CooCMatrix(shape, dest, b.rowIndices.clone(), b.colIndices.clone());
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    public CMatrix div(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemDiv(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and data as this matrix.
     */
    @Override
    public CTensor toTensor() {
        return new CTensor(shape, data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must be broadcastable to the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    @Override
    public CTensor toTensor(Shape newShape) {
        ValidateParameters.ensureBroadcastable(shape, newShape);
        return new CTensor(newShape, data.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape)
     * @see #I(int, int)
     */
    public static CMatrix I(int size) {
        return I(size, size);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int)
     * @see #I(Shape)
     */
    public static CMatrix I(int numRows, int numCols) {
        ValidateParameters.ensureNonNegative(numRows, numCols);
        Complex128[] entries = new Complex128[numRows*numCols];
        Arrays.fill(entries, Complex128.ZERO);
        int stop = Math.min(numRows, numCols);

        for(int i=0; i<stop; i++)
            entries[i*numCols+i] = Complex128.ONE;

        return new CMatrix(new Shape(numRows, numCols), entries);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape Shape of the identity-like matrix.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified shape is not rank 2.
     * @see #I(int)
     * @see #I(int, int)
     */
    public static CMatrix I(Shape shape) {
        ValidateParameters.ensureRank(shape, 2);
        return I(shape.get(0), shape.get(1));
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.
     *
     * <p>For large {@code n} values, this method <i>may</i> be significantly more efficient than calling
     * {@link #mult(Matrix) this.mult(this)} a total of {@code n} times.
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     *
     * @throws IllegalArgumentException If this matrix is not square (i.e. {@code !this.isSquare()}).
     */
    public CMatrix pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return CMatrix.I(numRows);
        if (n == 1) return copy();
        if (n == 2) return mult(this);

        CMatrix result = CMatrix.I(numRows);  // Start with identity matrix.
        CMatrix base = this;

        // Compute the matrix power efficiently using an "exponentiation by squaring" approach.
        while(n > 0) {
            if((n & 1) == 1)  // If n is odd.
                result = result.mult(base);

            base = base.mult(base);  // Square the base.
            n >>= 1;  // Divide n by 2 (bitwise right shift).
        }

        return result;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(Matrix b) {
        Complex128[] dest = MatrixMultiplyDispatcher.dispatch(this, b);
        return makeLikeTensor(new Shape(numRows, b.numCols), dest);
    }


    /**
     * Computes the matrix-vector multiplication between this matrix and a vector.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of length of the vector {@code b}.
     */
    public CVector mult(Vector b) {
        return MatrixMultiplyDispatcher.dispatch(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrMatrix b) {
        return (CMatrix) RealFieldDenseCsrMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CooMatrix b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows*b.numCols];
        RealFieldDenseCooMatMult.standard(
                data, shape, b.data, b.rowIndices, b.colIndices, b.shape, dest);
        Shape shape = new Shape(this.numRows, b.numCols);

        return new CMatrix(shape, dest);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrCMatrix b) {
        return (CMatrix) DenseCsrFieldMatMult .standard(this, b);
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
    public CMatrix mult(CooCMatrix b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows*b.numCols];
        DenseCooFieldMatMult.standard(
                data, shape, b.data, b.rowIndices, b.colIndices, b.shape, dest);
        Shape shape = new Shape(numRows, b.numCols);

        return new CMatrix(shape, dest);
    }


    /**
     * Computes the matrix-vector multiplication between this matrix and a vector.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of length of the vector {@code b}.
     */
    public CVector mult(CooVector b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows];
        RealFieldDenseCooMatMult.blockedVector(data, shape, b.data, b.indices, dest);
        return new CVector(dest);
    }


    /**
     * Computes the matrix-vector multiplication between this matrix and a vector.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of length of the vector {@code b}.
     */
    public CVector mult(CooCVector b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows];
        DenseCooFieldMatMult.blockedVector(data, shape, b.data, b.indices, dest);
        return new CVector(dest);
    }


    /**
     * Converts this complex matrix to a real matrix. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real matrix containing the real components of the data of this matrix.
     */
    public Matrix toReal() {
        return new Matrix(shape, Complex128Ops.toReal(data));
    }


    /**
     * Checks if all data of this matrix are real.
     * @return {@code true} if all data of this matrix are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this matrix has non-zero imaginary component.
     * @return {@code true} if any entry of this matrix has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is approximately equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix it is unitary; {@code false} otherwise.
     */
    public boolean isUnitary() {
        // TODO: Investigate what precision should be used in rounding.
        return numRows == numCols && mult(H()).round(8).isI();
    }


    /**
     * Rounds all data in this matrix to the nearest integer. The real and imaginary components will be rounded
     * independently.
     * @return A new matrix containing the data of this matrix rounded to the nearest integer.
     */
    public CMatrix round() {
        return round(0);
    }


    /**
     * Rounds all data within this matrix to the specified precision. The real and imaginary components will be rounded
     * independently.
     * @param precision The precision to round to (i.e. the number of decimal places to round to). Must be non-negative.
     * @return A new matrix containing the data of this matrix rounded to the specified precision.
     */
    public CMatrix round(int precision) {
        return new CMatrix(shape, Complex128Ops.round(data, precision));
    }


    /**
     * Sets all elements of this matrix to zero if they are within {@code tol} of zero. This is <i>not</i> done in place.
     * @param precision The precision to round to (i.e. the number of decimal places to round to). Must be non-negative.
     * @return A copy of this matrix with all data within {@code tol} of zero set to zero.
     */
    public CMatrix roundToZero(double tolerance) {
        return new CMatrix(shape, Complex128Ops.roundToZero(data, tolerance));
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type {@link CMatrix}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CMatrix src2 = (CMatrix) object;

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Gets a row of the matrix formatted as a human-readable string.
     * @param rowIndex Index of the row to get.
     * @param columnsToPrint List of column indices to print.
     * @param maxWidths List of maximum string lengths for each column.
     * @return A human-readable string representation of the specified row.
     */
    private String rowToString(int rowIndex, List<Integer> columnsToPrint, List<Integer> maxWidths) {
        StringBuilder sb = new StringBuilder();

        // Start the row with appropriate bracket.
        sb.append(rowIndex > 0 ? " [" : "[");

        // Loop over the columns to print.
        for (int i = 0; i < columnsToPrint.size(); i++) {
            int colIndex = columnsToPrint.get(i);
            String value;
            int width = PrintOptions.getPadding() + maxWidths.get(i);

            if (colIndex == -1) // Placeholder for truncated columns.
                value = "...";
            else
                value = StringUtils.ValueOfRound(this.get(rowIndex, colIndex), PrintOptions.getPrecision());

            if (PrintOptions.useCentering())
                value = StringUtils.center(value, width);

            sb.append(String.format("%-" + width + "s", value));
        }

        // Close the row.
        sb.append("]");

        return sb.toString();
    }


    /**
     * Generates a human-readable string representing this matrix.
     * @return A human-readable string representing this matrix.
     */
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        if (data.length == 0) {
            result.append("[]"); // No data in this matrix.
        } else {
            int numRows = this.numRows;
            int numCols = this.numCols;

            int maxRows = PrintOptions.getMaxRows();
            int maxCols = PrintOptions.getMaxColumns();

            int rowStopIndex = Math.min(maxRows - 1, numRows - 1);
            boolean truncatedRows = maxRows < numRows;

            int colStopIndex = Math.min(maxCols - 1, numCols - 1);
            boolean truncatedCols = maxCols < numCols;

            // Build list of column indices to print
            List<Integer> columnsToPrint = new ArrayList<>();
            for (int j = 0; j < colStopIndex; j++)
                columnsToPrint.add(j);

            if (truncatedCols) columnsToPrint.add(-1); // Use -1 to indicate '...'.
            columnsToPrint.add(numCols - 1); // Always include the last column.

            // Compute maximum widths for each column
            List<Integer> maxWidths = new ArrayList<>();
            for (Integer colIndex : columnsToPrint) {
                int maxWidth;
                if (colIndex == -1)
                    maxWidth = 3; // Width for '...'.
                else
                    maxWidth = PrettyPrint.maxStringLength(this.getCol(colIndex).data, rowStopIndex + 1);

                maxWidths.add(maxWidth);
            }

            // Build the rows up to the stopping index.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(i, columnsToPrint, maxWidths));
                result.append("\n");
            }

            if (truncatedRows) {
                // Print a '...' row to indicate truncated rows.
                int totalWidth = maxWidths.stream().mapToInt(w -> w + PrintOptions.getPadding()).sum();
                String value = "...";

                if (PrintOptions.useCentering())
                    value = StringUtils.center(value, totalWidth);

                result.append(String.format(" [%-" + totalWidth + "s]\n", value));
            }

            // Append the last row.
            result.append(rowToString(numRows - 1, columnsToPrint, maxWidths));
        }

        result.append("]");

        return result.toString();
    }
}
