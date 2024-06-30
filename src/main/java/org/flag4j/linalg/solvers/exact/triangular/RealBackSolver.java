/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.solvers.exact.triangular;

import org.flag4j.dense.Matrix;
import org.flag4j.dense.Vector;
import org.flag4j.util.exceptions.SingularMatrixException;


/**
 * This solver solves linear systems of equations where the coefficient matrix in an {@link Matrix#isTriU() upper triangular} real dense matrix
 * and the constant vector is a real dense vector or matrix. That is, solves a linear system of equations {@code U*x=b} or
 * {@code U*X=B} where {@code U} is an upper triangular matrix.
 */
public class RealBackSolver extends BackSolver<Matrix, Vector, double[]> {

    /**
     * For computing determinant of coefficient matrix during solve.
     */
    protected double det;

    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices. Note, by default no check will
     *      * be made to ensure the coefficient matrix is upper triangular. If you would like to enforce this, see
     *      * {@link #RealBackSolver(boolean)}.
     */
    public RealBackSolver() {
        super(false);
    }


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public RealBackSolver(boolean enforceTriU) {
        super(enforceTriU);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public double getDet() {
        return det;
    }


    /**
     * Solves the linear system of equations given by {@code U*x=b} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *          this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    @Override
    public Vector solve(Matrix U, Vector b) {
        checkParams(U, b.size);

        double sum, diag;
        int uIndex;
        int n = b.size;
        x = new Vector(U.numRows);
        det = 1;

        x.entries[n-1] = b.entries[n-1]/U.entries[n*n-1];

        for(int i=n-2; i>-1; i--) {
            sum = 0;
            uIndex = i*U.numCols;

            diag = U.entries[i*(n+1)];
            det*=diag;

            for(int j=i+1; j<n; j++) {
                sum += U.entries[uIndex + j]*x.entries[j];
            }

            x.entries[i] = (b.entries[i]-sum)/diag;
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return x;
    }


    /**
     * Solves the linear system of equations given by {@code U*X=B} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code U*X=B}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    @Override
    public Matrix solve(Matrix U, Matrix B) {
        checkParams(U, B.numRows);

        double sum, diag;
        int uIndex, xIndex;
        int n = B.numRows;
        double uValue = U.entries[n*n-1];
        int rowOffset = (n-1)*B.numCols;
        X = new Matrix(B.shape);
        det = 1;

        xCol = new double[n];

        for(int j=0; j<B.numCols; j++) {
            X.entries[rowOffset + j] = B.entries[rowOffset + j]/uValue;

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=n-2; i>=0; i--) {
                sum = 0;
                uIndex = i*U.numCols;
                xIndex = i*X.numCols + j;
                diag = U.entries[i*(n+1)];

                det*=diag;

                for(int k=i+1; k<n; k++) {
                    sum += U.entries[uIndex + k]*xCol[k];
                }

                double value = (B.entries[xIndex] - sum) / diag;

                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves the linear system of equations given by {@code U*X=I} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix and {@code I} is the {@link Matrix#isI() identity}
     * matrix of appropriate size. This essentially inverts the upper triangular matrix since {@code U*U}<sup>-1</sup>{@code =I}.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @return The solution to {@code X} in the linear system {@code U*X=B}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    public Matrix solveIdentity(Matrix U) {
        checkParams(U, U.numRows);

        double sum, diag;
        int uIndex, xIndex;
        int n = U.numRows;
        X = new Matrix(U.shape);
        det = U.entries[n*n-1];

        xCol = new double[n];
        X.entries[X.entries.length-1] = 1.0/det;

        for(int j=0; j<U.numCols; j++) {
            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=n-2; i>-1; i--) {
                sum = (i == j) ? 1 : 0;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                uIndex += i+1;
                diag = U.entries[i*(n+1)];

                det*=diag;

                for(int k=i+1; k<n; k++) {
                    sum -= U.entries[uIndex++]*xCol[k];
                }

                double value = sum / diag;
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves a special case of the linear system {@code U*X=L} for {@code X} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix and the constant matrix {@code L} is
     * {@link Matrix#isTriL() lower triangular}.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code enforceTriU} was set to {@code false} when
     *      this solver instance was created and {@code U} is not actually upper triangular, it will be treated as if it were.
     * @param L Lower triangular constant matrix. This is not explicit checked. If {@code L} is not lower triangular, values above
     *          the principle diagonal will be ignored and the result will still be correctly computed.
     * @return The result of solving the linear system {@code U*X=L} for the matrix {@code X}.
     */
    public Matrix solveLower(Matrix U, Matrix L) {
        checkParams(U, L.numRows);

        double sum, diag;
        int uIndex, xIndex;
        int n = L.numRows;
        double uValue = U.entries[U.entries.length-1];
        int rowOffset = (n-1)*n;
        X = new Matrix(L.shape);
        det = uValue;

        xCol = new double[n];

        for(int j=0; j<n; j++) {
            X.entries[rowOffset] = L.entries[rowOffset++]/uValue;

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=L.numCols-2; i>=0; i--) {
                sum = 0;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;

                diag = U.entries[i*(n+1)];
                det*=diag;

                for(int k=i+1; k<n; k++) {
                    sum += U.entries[uIndex + k]*xCol[k];
                }

                double value = (L.entries[xIndex] - sum) / diag;
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }
}
