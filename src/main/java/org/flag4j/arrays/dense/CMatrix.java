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
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.backend.TensorBase;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.operations_old.MatrixMultiplyDispatcher;
import org.flag4j.util.ParameterChecks;

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
    public CMatrix(Shape shape, Complex128[] entries) {
        super(shape, entries);
        ParameterChecks.ensureRank(shape, 2);
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
    public CMatrix(int rows, int cols, Complex128[] entries) {
        super(new Shape(rows, cols), entries);
        ParameterChecks.ensureRank(shape, 2);
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
    public CMatrix(Complex128[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(Complex128[] row : entries) {
            for(Complex128 value : row)
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
    public CMatrix makeLikeTensor(Shape shape, Complex128[] entries) {
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
    public CVector makeLikeVector(Complex128... entries) {
        return new CVector(entries);
    }


    /**
     * Converts this complex matrix to a real matrix. This is done by ignoring the imaginary component of all entries.
     * @return A real matrix containing the real components of this complex matrices entries.
     */
    public Matrix toReal() {
        double[] real = new double[entries.length];
        for(int i=0, size=entries.length; i<size; i++)
            real[i] = entries[i].re;

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
        List<Complex128> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                Complex128 val = entries[rowOffset + j];

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
        return new CMatrix(new Shape(numRows, numCols));
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
        ParameterChecks.ensureRank(shape, 2);
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
}
