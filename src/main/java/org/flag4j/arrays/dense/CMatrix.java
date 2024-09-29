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

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.backend.TensorBase;
import org.flag4j.arrays.sparse.*;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.operations.MatrixMultiplyDispatcher;
import org.flag4j.operations.common.complex.Complex128Operations;
import org.flag4j.operations.common.complex.ComplexProperties;
import org.flag4j.operations.dense.complex.ComplexDenseEquals;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseMatrixMultiplication;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseMatrixOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixOperations;
import org.flag4j.operations.dense_sparse.csr.complex.ComplexCsrDenseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.complex.ComplexCsrDenseOperations;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseMatrixMultiplication;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A complex dense matrix backed by a {@link Complex128} array.</p>
 *
 * <p>A CMatrix has mutable entries but fixed shape.</p>
 *
 * <p>A matrix is essentially equivalent to a rank 2 tensor but has some extended functionality and <i>may</i> have improved performance
 * for some operations.</p>
 */
public class CMatrix extends DenseFieldMatrixBase<CMatrix, CooCMatrix, CsrCMatrix, CVector, Complex128> {

    /**
     * Creates a complex matrix with the specified {@code entries} and {@code shape}.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(Shape shape, Field<Complex128>[] entries) {
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
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a zero matrix with the specified {@code shape}.
     *
     * @param shape Shape of this matrix.
     */
    public CMatrix(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Creates a square zero matrix with the specified {@code size}.
     *
     * @param size Size of the zero matrix to construct. The resulting matrix will have shape {@code (size, size)}
     */
    public CMatrix(int size) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code entries}, and shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(int rows, int cols, Field<Complex128>[] entries) {
        super(new Shape(rows, cols), entries);
        ValidateParameters.ensureRank(shape, 2);
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
        Arrays.fill(entries, fillValue);
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
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Constructs a complex matrix from a 2D array. The matrix will have the same shape as the array.
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(Field<Complex128>[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(Field<Complex128>[] row : entries) {
            for(Field<Complex128> value : row)
                super.entries[flatPos++] = value;
        }
    }


    /**
     * <p>Constructs a complex matrix from a 2D array of strings. Each string must be formatted properly as a complex number that can
     * be parsed by {@link org.flag4j.io.parsing.ComplexNumberParser}</p>
     *
     * <p>The matrix will have the same shape as the array.</p>
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(String[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(String[] row : entries) {
            for(String value : row)
                super.entries[flatPos++] = new Complex128(value);
        }
    }


    /**
     * Constructs a complex matrix with specified {@code shape} and {@code entries}.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix.
     */
    public CMatrix(Shape shape, double[] entries) {
        super(shape, new Complex128[entries.length]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
        ArrayUtils.arraycopy(entries, 0, super.entries, 0, entries.length);
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
                super.entries[idx++] = new Complex128(value);
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
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Complex128 fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Double fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates matrix with the specified {@code shape} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(Shape shape, Double fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Constructs a copy of the specified matrix.
     * @param mat Matrix to create copy of.
     */
    public CMatrix(CMatrix mat) {
        super(mat.shape, mat.entries.clone());
    }


    /**
     * Constructs an empty complex matrix with the specified shape. The entries of this matrix will be {@code null}.
     * @param numRows The number of rows in the matrix.
     * @param numCols The number of columns in the matrix.
     * @return An empty complex matrix with the specified shape.
     */
    public static CMatrix getEmpty(int numRows, int numCols) {
        return new CMatrix(new Shape(numRows, numCols), new Complex128[numRows*numCols]);
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
    public CMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a matrix of the same type as this matrix with the given the shape filled with the specified fill value.
     *
     * @param shape Shape of the matrix to construct.
     * @param fillValue Value to fill this matrix with.
     *
     * @return A matrix of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CMatrix makeLikeTensor(Shape shape, Complex128 fillValue) {
        return new CMatrix(shape, fillValue);
    }


    /**
     * Constructs a vector of similar type to this matrix with the given {@code entries}.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of similar type to this matrix with the given {@code entries}.
     */
    @Override
    public CVector makeLikeVector(Field<Complex128>... entries) {
        return new CVector(entries);
    }


    /**
     * Converts this complex matrix to a real matrix. This is done by ignoring the imaginary component of all entries.
     * @return A real matrix containing the real components of this complex matrices entries.
     */
    public Matrix toReal() {
        double[] real = new double[entries.length];
        for(int i=0, size=entries.length; i<size; i++)
            real[i] = ((Complex128) entries[i]).re;

        return new Matrix(shape, real);
    }


    /**
     * Converts this dense matrix to an equivalent compressed sparse row (CSR) matrix.
     *
     * @return A CSR matrix equivalent to this matrix.
     */
    @Override
    public CsrCMatrix toCsr() {
        return toCoo().toCsr();
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooCMatrix toCoo() {
        int rows = numRows;
        int cols = numCols;
        List<Field<Complex128>> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                Field<Complex128> val = entries[rowOffset + j];

                if(!val.isZero()) {
                    sparseEntries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooCMatrix(shape, sparseEntries, rowIndices, colIndices);
    }


    /**
     * Constructs an identity matrix of the specified size.
     * @param size The size of the identity matrix to construct.
     * @return An identity matrix of shape {@code (size, size)}.
     */
    public static CMatrix I(int size) {
        return I(new Shape(size, size));
    }


    /**
     * Constructs an identity matrix with the specified shape. If the specified shape is not square, then a rectangular matrix with
     * ones along the principle diagonal will be created.
     * @param numRows Number of rows in the identity matrix.
     * @param numCols Number of columns in the identity matrix.
     * @return An identity matrix with the specified shape. If the specified shape is not square, then a rectangular matrix with
     * ones along the principle diagonal will be created.
     * @see #I(Shape) 
     */
    public static CMatrix I(int numRows, int numCols) {
        return I(new Shape(numRows, numCols));
    }


    /**
     * Constructs an identity matrix with the specified shape. If the specified shape is not square, then a rectangular matrix with
     * ones along the principle diagonal will be created.
     * @param numRows Number of rows in the identity matrix.
     * @param numCols Number of columns in the identity matrix.
     * @return An identity matrix with the specified shape. If the specified shape is not square, then a rectangular matrix with
     * ones along the principle diagonal will be created.
     * @see #I(int, int)
     */
    public static CMatrix I(Shape shape) {
        ValidateParameters.ensureRank(shape, 2);
        CMatrix I = new CMatrix(shape);
        final int stop = I.entries.length;
        final int step = I.numCols + 1;

        for(int i=0; i<stop; i+=step)
            I.entries[i] = Complex128.ONE;

        return I;
    }


    /**
     * <p>Computes the rank of this matrix (i.e. the number of linearly independent rows/columns in this matrix).</p>
     *
     * <p>This is computed as the number of singular values greater than {@code tol} where:
     * <pre>{@code double tol = 2.0*Math.max(rows, cols)*Flag4jConstants.EPS_F64*Math.min(this.numRows, this.numCols);}</pre>
     * </p>
     *
     * <p>Note the "matrix rank" is <b>NOT</b> related to the "{@link TensorBase#getRank() tensor rank}" which is number of indices
     * needed to uniquely specify an entry in the tensor.</p>
     *
     * @return The matrix rank of this matrix.
     */
    public int matrixRank() {
        return new ComplexSVD(false).decompose(this).getRank();
    }


    /**
     * Multiplies this complex dense matrix with a real dense matrix.
     * @param b The second matrix in the matrix multiplication.
     * @return The matrix product of this matrix and {@code b}.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code this.numCols != b.numRows}
     */
    public CMatrix mult(Matrix b) {
        Complex128[] entries = MatrixMultiplyDispatcher.dispatch(this, b);
        Shape shape = new Shape(this.numRows, b.numCols);

        return new CMatrix(shape, entries);
    }


    /**
     * Checks if this matrix is unitary. That is, if this matrices inverse is equal to its Hermitian transpose.
     *
     * @return True if this matrix it is unitary. Otherwise, returns false.
     */
    public boolean isUnitary() {
        if(isSquare()) {
            return mult(H()).isCloseToI();
        } else {
            return false;
        }
    }


    /**
     * Sums this matrix with a real dense matrix.
     * @param b Real dense matrix in the sum.
     * @return The element-wise sum of this matrix with {@code b}.
     */
    public CMatrix add(Matrix b) {
        return new CMatrix(this.shape, RealComplexDenseOperations.add(this.entries, this.shape, b.entries, b.shape));
    }


    /**
     * Sums this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public CMatrix add(CsrMatrix b) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, b, Complex128::add);
    }


    /**
     * Sums this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public CMatrix add(CooMatrix b) {
        return RealComplexDenseSparseMatrixOperations.add(this, b);
    }


    /**
     * Sums this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public CMatrix add(CsrCMatrix b) {
        return ComplexCsrDenseOperations.applyBinOpp(this, b, (Complex128 x, Complex128 y) -> x.add(y));
    }


    /**
     * Sums this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public CMatrix add(CooCMatrix b) {
        return ComplexDenseSparseMatrixOperations.add(this, b);
    }


    /**
     * Computes the difference of this matrix with a real dense matrix.
     * @param b Real dense matrix in the difference.
     * @return The difference of this matrix with {@code b}.
     */
    public CMatrix sub(Matrix b) {
        return new CMatrix(this.shape,
                RealComplexDenseOperations.sub(this.entries, this.shape, b.entries, b.shape)
        );
    }


    /**
     * Computes the difference of this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public CMatrix sub(CsrMatrix b) {
        return RealComplexCsrDenseOperations.applyBinOpp(this, b, Complex128::sub);
    }


    /**
     * Computes the difference of this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public CMatrix sub(CooMatrix b) {
        return RealComplexDenseSparseMatrixOperations.sub(this, b);
    }


    /**
     * Computes the difference of this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public CMatrix sub(CsrCMatrix b) {
        return ComplexCsrDenseOperations.applyBinOpp(this, b, Complex128::sub);
    }


    /**
     * Computes the difference of this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public CMatrix sub(CooCMatrix b) {
        return ComplexDenseSparseMatrixOperations.add(this, b);
    }


    /**
     * Computes the matrix multiplication between this matrix and a real sparse CSR matrix.
     * @param b The real sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrMatrix b) {
        return RealComplexCsrDenseMatrixMultiplication.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between this matrix and a complex sparse CSR matrix.
     * @param b The complex sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrCMatrix b) {
        return ComplexCsrDenseMatrixMultiplication.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between this matrix and a real sparse COO matrix.
     * @param b The real sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code this.numCols != b.numRows}.
     * @implNote This method computes the matrix product as {@code this.mult(b.toCsr());}.
     */
    public CMatrix mult(CooMatrix b) {
        return mult(b.toCsr());
    }


    /**
     * Computes the matrix multiplication between this matrix and a complex sparse COO matrix.
     * @param b The complex sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code this.numCols != b.numRows}.
     * @implNote This method computes the matrix product as {@code this.mult(b.toCsr());}.
     */
    public CMatrix mult(CooCMatrix b) {
        return mult(b.toCsr());
    }


    /**
     * Computes the matrix-vector product of this matrix and a real dense vector.
     * @param b Vector in the matrix-vector product.
     * @return The matrix-vector product of this matrix and the vector {@code b}.
     */
    public CVector mult(Vector b) {
        ValidateParameters.ensureMatMultShapes(this.shape, new Shape(b.size, 1));
        Complex128[] entries = RealComplexDenseMatrixMultiplication.standardVector(
                this.entries, this.shape, b.entries, b.shape
        );

        return new CVector(entries);
    }


    /**
     * Computes the matrix-vector product of this matrix and a real sparse vector.
     * @param b Vector in the matrix-vector product.
     * @return The matrix-vector product of this matrix and the vector {@code b}.
     */
    public CVector mult(CooVector b) {
        ValidateParameters.ensureMatMultShapes(this.shape, new Shape(b.size, 1));
        Complex128[] entries = RealComplexDenseSparseMatrixMultiplication.standardVector(
                this.entries, this.shape, b.entries, b.indices
        );

        return new CVector(entries);
    }


    /**
     * Computes the matrix-vector product of this matrix and a complex sparse vector.
     * @param b Vector in the matrix-vector product.
     * @return The matrix-vector product of this matrix and the vector {@code b}.
     */
    public CVector mult(CooCVector b) {
        ValidateParameters.ensureMatMultShapes(this.shape, new Shape(b.size, 1));
        Complex128[] entries = ComplexDenseSparseMatrixMultiplication.standardVector(
                this.entries, this.shape, b.entries, b.indices
        );


        return new CVector(entries);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     * @return A tensor equivalent to this matrix.
     */
    public CTensor toTensor() {
        return new CTensor(shape, entries);
    }


    /**
     * Rounds this tensor to the nearest whole number. If the tensor is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @return A copy of this tensor with each entry rounded to the nearest whole number.
     */
    public CMatrix round() {
        return round(0);
    }


    /**
     * Rounds a matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     */
    public CMatrix round(int precision) {
        return new CMatrix(this.shape, Complex128Operations.round(this.entries, precision));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the tensor is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within {@link Flag4jConstants#EPS_F64} of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this tensor with rounded values.
     */
    public CMatrix roundToZero() {
        return roundToZero(Flag4jConstants.EPS_F64);
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     */
    public CMatrix roundToZero(double threshold) {
        return new CMatrix(this.shape, Complex128Operations.roundToZero(this.entries, threshold));
    }


    /**
     * Checks if this complex matrix only contains real values.
     * @return True if every entry in this matrix has zero imaginary component. False otherwise.
     */
    public boolean isReal() {
        return ComplexProperties.isReal(entries);
    }


    /**
     * Checks if this complex matrix contains at least one non-real value.
     * @return True if at least one value in this matrix is non-real. False otherwise.
     */
    public boolean isComplex() {
        return ComplexProperties.isComplex(entries);
    }


    /**
     * Sets the values of this matrix to {@code values}.
     * @param values New values of the array. Must have shape {@code new double[this.numRows][this.numCols]}.
     * @throws IllegalArgumentException If {@code values.length != this.numRows || values[0].length != this.numCols}.
     */
    public void setValues(double[][] values) {
        ValidateParameters.ensureEquals(numRows, values.length);
        ValidateParameters.ensureEquals(numCols, values.length);

        int idx = 0;
        for(int i=0; i<numRows; i++) {
            for(int j=0; j<numCols; j++) {
                entries[idx++] = new Complex128(values[i][j]);
            }
        }
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
    public CMatrix setSlice(Matrix values, int rowStart, int colStart) {
        ValidateParameters.ensureValidIndices(numRows, rowStart);
        ValidateParameters.ensureValidIndices(numCols, colStart);

        CMatrix copy = copy();

        for(int i=0, rows=values.numRows; i<rows; i++) {
            int copyOffset = (i+rowStart)*numCols + colStart;
            int valuesRowOffset = i*values.numCols;

            for(int j=0, cols=values.numCols; j<cols; j++) {
                copy.entries[copyOffset + j] = new Complex128(values.entries[valuesRowOffset + j]);
            }
        }

        return copy;
    }


    /**
     * Creates a zero matrix with the specified shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     */
    public CMatrix div(Matrix b) {
        return new CMatrix(
                shape,
                RealComplexDenseElemDiv.dispatch(entries, shape, b.entries, b.shape)
        );
    }


    /**
     * Computes the element-wise product between this matrix and a real dense matrix.
     * @param b Second matrix in the element-wise product.
     * @return
     */
    public CMatrix elemMult(Matrix b) {
        return new CMatrix(
                shape,
                RealComplexDenseElemMult.dispatch(entries, shape, b.entries, b.shape)
        );
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
        return RealComplexDenseSparseMatrixOperations.elemMult(this, b);
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
        return ComplexDenseSparseMatrixOperations.elemMult(this, b);
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.</p>
     *
     * <p>For large {@code n} values, this method <i>may</i> significantly more efficient than calling
     * {@link #mult(CMatrix) this.mult(this)} a total of {@code n} times.</p>
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public CMatrix pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return CMatrix.I(numRows);
        if (n == 1) return copy();
        if (n == 2) return mult(this);

        CMatrix result = CMatrix.I(numRows);  // Start with identity matrix
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
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this vector.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link Matrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CMatrix src2 = (CMatrix) object;

        return ComplexDenseEquals.tensorEquals(entries, shape, src2.entries, src2.shape);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * Gets row of matrix formatted as a human-readable String. Helper method for {@link #toString} method.
     * @param i Index of row to get.
     * @param colStopIndex Stopping index for printing columns.
     * @param maxList List of maximum string representation lengths for each column of this matrix. This
     *                is used to align columns when printing.
     * @return A human-readable String representation of the specified row.
     */
    private String rowToString(int i, int colStopIndex, List<Integer> maxList) {
        int width;
        String value;
        StringBuilder result = new StringBuilder();

        if(i>0) {
            result.append(" [");
        }  else {
            result.append("[");
        }

        for(int j=0; j<colStopIndex; j++) {
            value = StringUtils.ValueOfRound(this.get(i, j), PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + maxList.get(j);
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(PrintOptions.getMaxColumns() < this.numCols) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry in the column now
        value = StringUtils.ValueOfRound(this.get(i, this.numCols-1), PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + maxList.getLast();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s]", value));

        return result.toString();
    }


    /**
     * Generates a human-readable string representing this matrix.
     * @return A human-readable string representing this matrix.
     */
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");

        result.append("[");

        if(entries.length==0) {
            result.append("[]"); // No entries in this matrix.
        } else {
            int rowStopIndex = Math.min(PrintOptions.getMaxRows() - 1, this.numRows - 1);
            int colStopIndex = Math.min(PrintOptions.getMaxColumns() - 1, this.numCols - 1);
            int width;
            int totalRowLength = 0; // Total string length of each row (not including brackets)
            String value;

            // Find maximum entry string width in each column so columns can be aligned.
            List<Integer> maxList = new ArrayList<>(colStopIndex + 1);
            for (int j = 0; j < colStopIndex; j++) {
                maxList.add(PrettyPrint.maxStringLength(this.getCol(j).entries, rowStopIndex));
                totalRowLength += maxList.getLast();
            }

            if (colStopIndex < this.numCols) {
                maxList.add(PrettyPrint.maxStringLength(this.getCol(this.numCols - 1).entries));
                totalRowLength += maxList.getLast();
            }

            if (colStopIndex < this.numCols - 1) {
                totalRowLength += 3 + PrintOptions.getPadding(); // Account for '...' element with padding in each column.
            }

            totalRowLength += maxList.size() * PrintOptions.getPadding(); // Account for column padding

            // Get each row as a string.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(i, colStopIndex, maxList));
                result.append("\n");
            }

            if (PrintOptions.getMaxRows() < this.numRows) {
                width = totalRowLength;
                value = "...";
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format(" [%-" + width + "s]\n", value));
            }

            // Get Last row as a string.
            result.append(rowToString(this.numRows - 1, colStopIndex, maxList));
        }

        result.append("]");

        return result.toString();
    }
}
