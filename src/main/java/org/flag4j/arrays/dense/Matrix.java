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
import org.flag4j.arrays.backend.DenseMatrixMixin;
import org.flag4j.arrays.backend.DensePrimitiveDoubleTensorBase;
import org.flag4j.arrays.backend.MatrixVectorOpsMixin;
import org.flag4j.arrays.backend.TensorBase;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.operations.MatrixMultiplyDispatcher;
import org.flag4j.operations.RealDenseMatrixMultiplyDispatcher;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.operations.dense.real.*;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A real dense matrix backed by a primitive double array.</p>
 *
 * <p>The {@link #entries} of a matrix are mutable but the {@link #shape} is fixed.</p>
 *
 *<p>A matrix is essentially equivalent to a rank 2 tensor but has some extended functionality and <i>may</i> have improved
 * performance for some operations.</p>
 */
public class Matrix extends DensePrimitiveDoubleTensorBase<Matrix, CooMatrix>
        implements DenseMatrixMixin<Matrix, CooMatrix, CsrMatrix, Double>,
        MatrixVectorOpsMixin<Matrix, Vector, Vector> {

    // TODO: Add dense/sparse and real/complex operations for selected operations.

    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public Matrix(Shape shape, double[] entries) {
        super(shape, entries);
        ParameterChecks.ensureRank(shape, 2);

        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Constructs a square real dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size) {
        super(new Shape(size, size), new double[size*size]);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a square real dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size, double value) {
        super(new Shape(size, size), new double[size*size]);
        Arrays.fill(super.entries, value);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols) {
        super(new Shape(rows, cols), new double[rows*cols]);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a real dense matrix with a specified shape and fills the matrix with the specified value.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols, double value) {
        super(new Shape(rows, cols), new double[rows*cols]);
        Arrays.fill(super.entries, value);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(Double[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);

        int index = 0;
        for(Double[] row : entries) {
            for(Double value : row) {
                super.entries[index++] = value;
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(Integer[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);

        int index = 0;
        for(Integer[] row : entries) {
            for(Integer value : row) {
                super.entries[index++] = value;
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(double[][] entries) {
        super(new Shape(entries.length, entries[0].length),
                new double[entries.length*entries[0].length]);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);

        int index = 0;
        for(double[] row : entries) {
            for(double value : row) {
                super.entries[index++] = value;
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(int[][] entries) {
        super(new Shape(entries.length, entries[0].length), new double[entries.length*entries[0].length]);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);

        // Copy the int array
        int index=0;
        for(int[] row : entries) {
            for(int value : row) {
                super.entries[index++] = value;
            }
        }
    }


    /**
     * Creates a real dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public Matrix(Matrix A) {
        super(A.shape, A.entries.clone());
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a real dense matrix with specified shape filled with zeros.
     * @param shape Shape of matrix.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape) {
        super(shape, new double[shape.totalEntries().intValue()]);
        ParameterChecks.ensureRank(shape, 2);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Creates a real dense matrix with specified shape filled with a specific value.
     * @param shape Shape of matrix.
     * @param value Value to fill matrix with.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape, double value) {
        super(shape, new double[shape.totalEntries().intValue()]);
        Arrays.fill(super.entries, value);
        ParameterChecks.ensureRank(shape, 2);
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);
    }


    /**
     * Constructs a matrix with specified shape and entries. Note, unlike other constructors, the entries' parameter
     * is not copied.
     * @param numRows Number of rows in this matrix.
     * @param numCols Number of columns in this matrix.
     * @param entries Entries of the matrix.
     */
    public Matrix(int numRows, int numCols, double[] entries) {
        super(new Shape(numRows, numCols), entries);
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
    public Matrix tensorDot(Matrix src2, int[] aAxes, int[] bAxes) {
        return RealDenseTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    public Matrix conj() {
        return copy();
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public Matrix H() {
        return T();
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public Matrix H(int axis1, int axis2) {
        return T(axis1, axis2);
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public Matrix H(int... axes) {
        return T(axes);
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
    public Matrix makeLikeTensor(Shape shape, double[] entries) {
        return new Matrix(shape, entries);
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
    public static Matrix I(int size) {
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
    public static Matrix I(int numRows, int numCols) {
        ParameterChecks.ensureNonNegative(numRows, numCols);
        Matrix I = new Matrix(numRows, numCols);
        int stop = Math.min(numRows, numCols);

        for(int i=0; i<stop; i++) {
            I.entries[i*numCols+i] = 1;
        }

        return I;
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
    public static Matrix I(Shape shape) {
        ParameterChecks.ensureRank(shape, 2);
        return I(shape.get(0), shape.get(1));
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
        ParameterChecks.ensureSquareMatrix(this.shape);
        double sum = 0;
        int colsOffset = this.numCols+1;

        for(int i=0; i<this.numRows; i++) {
            sum += this.entries[i*colsOffset];
        }

        return sum;
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

        // Ensure lower half is zeros.
        for(int i=1; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<i; j++) {
                if(entries[rowOffset + j] != 0) return false; // No need to continue.
            }
        }

        return true;
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

        // Ensure upper half is zeros.
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=i+1; j<numCols; j++) {
                if(entries[rowOffset + j] != 0) return false; // No need to continue.
            }
        }

        return true; // If we reach this point the matrix is lower triangular.
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     *
     * @return True if this matrix is singular or non-square. Otherwise, returns false.
     */
    @Override
    public boolean isSingular() {
        boolean result = true;

        if(isSquare()) {
            double tol = 1.0E-16; // Tolerance for determining if determinant is zero.
            double det = RealDenseDeterminant.det(this);

            result = Math.abs(det) < tol;
        }

        return result;
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
        return RealDenseProperties.isIdentity(this);
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
        return RealDenseProperties.isCloseToIdentity(this);
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    @Override
    public Double det() {
        return RealDenseDeterminant.det(this);
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
    @Override
    public int matrixRank() {
        return new RealSVD(false).decompose(this).getRank();
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix B.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(Matrix b) {
        double[] entries = RealDenseMatrixMultiplyDispatcher.dispatch(this, b);
        Shape shape = new Shape(this.numRows, b.numCols);

        return new Matrix(shape, entries);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose/
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public Matrix multTranspose(Matrix b) {
        // Ensure this matrix can be multiplied to the transpose of B.
        ParameterChecks.ensureEquals(this.numCols, b.numCols);

        return new Matrix(
                new Shape(this.numRows, b.numRows),
                RealDenseMatrixMultiplyDispatcher.dispatchTranspose(this, b)
        );
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(Matrix b) {
        ParameterChecks.ensureEqualShape(this.shape, b.shape);
        return this.T().mult(b).tr();
    }


    /**
     * Stacks matrices along columns.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(TensorBase, int)
     * @see #augment(Matrix)
     */
    @Override
    public Matrix stack(Matrix b) {
        ParameterChecks.ensureArrayLengthsEq(this.numCols, b.numCols);
        Matrix stacked = new Matrix(new Shape(this.numRows + b.numRows, this.numCols));

        System.arraycopy(this.entries, 0, stacked.entries, 0, this.entries.length);
        System.arraycopy(b.entries, 0, stacked.entries, this.entries.length, b.entries.length);

        return stacked;
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b MatrixOld to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(Matrix)
     * @see #stack(TensorBase, int)
     */
    @Override
    public Matrix augment(Matrix b) {
        ParameterChecks.ensureArrayLengthsEq(numRows, b.numRows);
        Matrix augmented = new Matrix(new Shape(numRows, numCols + b.numCols));

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augmented.entries, i*augmented.numCols, numCols);
        }

        // Copy entries from the B matrix.
        for(int i=0; i<b.numRows; i++) {
            int augmentedRowOffset = i*augmented.numCols;
            int bRowOffset = i*b.numCols;

            for(int j=0; j<b.numCols; j++) {
                augmented.entries[augmentedRowOffset + j + numCols] = b.entries[bRowOffset + j];
            }
        }

        return augmented;
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
    public Matrix swapRows(int rowIndex1, int rowIndex2) {
        ParameterChecks.ensureGreaterEq(0, rowIndex1, rowIndex2);
        ParameterChecks.ensureGreaterEq(rowIndex1, this.numRows-1);
        ParameterChecks.ensureGreaterEq(rowIndex2, this.numRows-1);

        double temp;
        int row1Start = rowIndex1*numCols;
        int row2Start = rowIndex2*numCols;
        int stop = row1Start + numCols;

        while(row1Start < stop) {
            temp = entries[row1Start];
            entries[row1Start++] = entries[row2Start];
            entries[row2Start++] = temp;
        }

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
    public Matrix swapCols(int colIndex1, int colIndex2) {
        ParameterChecks.ensureGreaterEq(0, colIndex1, colIndex2);
        ParameterChecks.ensureGreaterEq(colIndex1, this.numCols-1);
        ParameterChecks.ensureGreaterEq(colIndex2, this.numCols-1);

        double temp;
        for(int i=0; i<numRows; i++) {
            // Swap elements.
            int idx = i*numCols;
            ArrayUtils.swap(entries, idx + colIndex1, idx + colIndex1);
        }

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
        return RealDenseProperties.isSymmetric(entries, shape);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return isSymmetric();
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
        return RealDenseProperties.isAntiSymmetric(entries, shape);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        return numRows == numCols && RealDenseProperties.isCloseToIdentity(this.mult(this.T()));
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRow(int rowIndex) {
        Matrix copy = new Matrix(this.numRows-1, this.numCols);

        int row = 0;
        for(int i=0; i<this.numRows; i++) {
            if(i != rowIndex) {
                System.arraycopy(this.entries, i*numCols, copy.entries, row*copy.numCols, this.numCols);
                row++;
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRows(int... rowIndices) {
        Matrix copy = new Matrix(this.numRows-rowIndices.length, this.numCols);

        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(ArrayUtils.notContains(rowIndices, i)) {
                System.arraycopy(this.entries, i*numCols, copy.entries, row*copy.numCols, this.numCols);
                row++;
            }
        }

        return copy;
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCol(int colIndex) {
        Matrix copy = new Matrix(this.numRows, this.numCols-1);

        int col;
        for(int i=0; i<this.numRows; i++) {
            col = 0;
            for(int j=0; j<this.numCols; j++) {
                if(j!=colIndex) {
                    copy.entries[i*copy.numCols + col] = this.entries[i*numCols + j];
                    col++;
                }
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCols(int... colIndices) {
        Matrix copy = new Matrix(this.numRows, this.numCols - colIndices.length);

        int col;

        for(int i=0; i<this.numRows; i++) {
            col = 0;
            for(int j=0; j<this.numCols; j++) {
                if(ArrayUtils.notContains(colIndices, j)) {
                    copy.entries[i*copy.numCols + col] = this.entries[i*numCols + j];
                    col++;
                }
            }
        }

        return copy;
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
    public Matrix setSliceCopy(Matrix values, int rowStart, int colStart) {
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.numRows; i++) {
            System.arraycopy(
                    values.entries, i*values.numCols,
                    copy.entries, (i+rowStart)*numCols + colStart, values.numCols
            );
        }

        return copy;
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    @Override
    public Matrix setSlice(Matrix values, int rowStart, int colStart) {
        ParameterChecks.ensureLessEq(numRows, rowStart+values.numRows);
        ParameterChecks.ensureLessEq(numCols, colStart+values.numCols);
        ParameterChecks.ensureGreaterEq(0, rowStart, colStart);

        for(int i=0; i<values.numRows; i++) {
            System.arraycopy(
                    values.entries, i*values.numCols,
                    this.entries, (i+rowStart)*numCols + colStart, values.numCols
            );
        }

        return this;
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
    public Matrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        Matrix slice = new Matrix(rowEnd - rowStart, colEnd - colStart);

        for(int i=0; i<slice.numRows; i++) {
            System.arraycopy(
                    this.entries, (i+rowStart)*this.numCols + colStart,
                    slice.entries, i*slice.numCols, slice.numCols
            );
        }

        return slice;
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
    public Matrix set(Double value, int row, int col) {
        entries[row*numCols + col] = value;
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public Matrix setValues(Double[][] values) {
        ParameterChecks.ensureEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOperations.setValues(values, this.entries);
        return this;
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
    public Matrix setCol(Vector values, int colIndex) {
        ParameterChecks.ensureArrayLengthsEq(values.size, this.numRows);

        int rowOffset = 0;
        for(int i=0; i<values.size; i++) {
            super.entries[rowOffset + colIndex] = values.entries[i];
            rowOffset += numCols;
        }

        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     */
    @Override
    public Matrix setRow(Vector values, int rowIndex) {
        ParameterChecks.ensureArrayLengthsEq(values.size, this.numCols);
        int rowOffset = rowIndex*numCols;

        for(int i=0; i<values.size; i++)
            super.entries[rowOffset + i] = values.entries[i];

        return this;
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
    public Matrix getTriU(int diagOffset) {
        ParameterChecks.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        Matrix result = new Matrix(numRows, numCols);

        // Extract the upper triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=Math.max(0, i + diagOffset); j<numCols; j++) {
                if (j >= i + diagOffset) {
                    result.entries[rowOffset + j] = this.entries[rowOffset + j];
                }
            }
        }

        return result;
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
    public Matrix getTriL(int diagOffset) {
        ParameterChecks.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        Matrix result = new Matrix(numRows, numCols);

        // Extract the lower triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j <= Math.min(numCols - 1, i + diagOffset); j++) {
                if(j <= i + diagOffset) {
                    result.entries[rowOffset + j] = this.entries[rowOffset + j];
                }
            }
        }

        return result;
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
    public Vector mult(Vector b) {
        ParameterChecks.ensureMatMultShapes(this.shape, new Shape(b.size, 1));
        double[] entries = MatrixMultiplyDispatcher.dispatch(this, b);
        return new Vector(entries);
    }


    /**
     * <p>Converts this matrix to an equivalent vector.</p>
     *
     * <p>If this matrix is not shaped as a row/column vector, it will first be flattened then converted to a vector.</p>
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public Vector toVector() {
        return new Vector(entries.clone());
    }


    /**
     * Converts this matrix to an equivalent {@link Tensor}.
     * @return A {@link Tensor} which is equivalent to this matrix.
     */
    public Tensor toTensor() {
        return new Tensor(shape, entries.clone());
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
    public Vector getRow(int rowIdx) {
        ParameterChecks.ensureIndexInBounds(numRows, rowIdx);
        int start = rowIdx*numCols;
        int stop = start+numCols;

        double[] row = Arrays.copyOfRange(this.entries, start, stop);

        return new Vector(row);
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
    public Vector getRow(int rowIdx, int colStart, int colEnd) {
        ParameterChecks.ensureIndexInBounds(numCols, colStart, colEnd);
        ParameterChecks.ensureGreaterEq(colStart, colEnd);
        int start = rowIdx*numCols+colStart;
        int stop = start+colEnd;

        double[] row = Arrays.copyOfRange(this.entries, start, stop);

        return new Vector(row);
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
    public Vector getCol(int colIdx) {
        ParameterChecks.ensureValidIndices(numCols, colIdx);
        double[] col = new double[numRows];

        for(int i=0; i<numRows; i++)
            col[i] = entries[i*numCols + colIdx];

        return new Vector(col);
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
    public Vector getCol(int colIdx, int rowStart, int rowEnd) {
        ParameterChecks.ensureValidIndices(numRows, rowStart, rowEnd);
        ParameterChecks.ensureGreaterEq(rowEnd, rowStart);
        double[] col = new double[numRows];

        for(int i=rowStart; i<rowEnd; i++)
            col[i] = entries[i*numCols + colIdx];

        return new Vector(col);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public Vector getDiag() {
        int newSize = Math.min(numRows, numCols);
        double[] diag = new double[newSize];

        int idx = 0;
        for(int i=0; i<newSize; i++) {
            diag[i] = this.entries[idx];
            idx += numCols + 1;
        }

        return new Vector(diag);
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

        Matrix src2 = (Matrix) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
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
    public Matrix T(int axis1, int axis2) {
        ParameterChecks.ensureValidAxes(shape, axis1, axis2);

        if(axis1==axis2) return copy();
        else return TransposeDispatcher.dispatch(this);
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
    public Matrix T(int... axes) {
        ParameterChecks.ensureArrayLengthsEq(2, axes.length);
        ParameterChecks.ensureValidAxes(shape, axes[0], axes[1]);

        if(axes[0]==axes[1]) return copy();
        else return TransposeDispatcher.dispatch(this);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooMatrix toCoo() {
        int rows = numRows;
        int cols = numCols;
        List<Double> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                double val = entries[rowOffset + j];

                if(val != 0d) {
                    sparseEntries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooMatrix(shape, sparseEntries, rowIndices, colIndices);
    }


    /**
     * Converts this real dense matrix to an equivalent {@link CMatrix complex dense matrix}.
     * @return A complex dense matrix equivalent to this real dense matrix.
     */
    public CMatrix toComplex() {
        Complex128[] cmp = new Complex128[entries.length];
        for(int i=0, size=cmp.length; i<size; i++)
            cmp[i] = new Complex128(entries[i]); // Wrap value as complex number.

        return new CMatrix(shape, cmp);
    }
}
