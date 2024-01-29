/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.linalg.solvers;

import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.exceptions.SingularMatrixException;


/**
 * This solver solves linear systems of equations where the coefficient matrix in an {@link Matrix#isTriU() upper triangular} real dense matrix
 * and the constant vector is a real dense vector or matrix.
 */
public class RealBackSolver implements LinearSolver<Matrix, Vector> {

    /**
     * Solves the linear system of equations given by {@code U*x=b} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    @Override
    public Vector solve(Matrix U, Vector b) {
        double sum, diag;
        int uIndex;
        int n = b.size;
        Vector x = new Vector(U.numRows);

        x.entries[n-1] = b.entries[n-1]/U.entries[n*n-1];

        for(int i=n-2; i>-1; i--) {
            sum = 0;
            uIndex = i*U.numCols;

            diag = U.entries[i*(n+1)];
            if(diag==0) {
                throw new SingularMatrixException("Cannot solve linear system.");
            }

            for(int j=i+1; j<n; j++) {
                sum += U.entries[uIndex + j]*x.entries[j];
            }

            x.entries[i] = (b.entries[i]-sum)/diag;
        }

        return x;
    }


    /**
     * Solves the linear system of equations given by {@code U*X=B} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code U*X=B}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    @Override
    public Matrix solve(Matrix U, Matrix B) {
        double sum, diag;
        int uIndex, xIndex;
        int n = B.numRows;
        double uValue = U.entries[n*n-1];
        int rowOffset = (n-1)*B.numCols;
        Matrix X = new Matrix(B.shape);

        for(int j=0; j<B.numCols; j++) {
            X.entries[rowOffset + j] = B.entries[rowOffset + j]/uValue;

            for(int i=n-2; i>-1; i--) {
                sum = 0;
                uIndex = i*U.numCols;
                xIndex = i*X.numCols + j;
                diag = U.entries[i*(n+1)];

                if(diag==0) throw new SingularMatrixException("Cannot solve linear system.");

                for(int k=i+1; k<n; k++) {
                    sum += U.entries[uIndex + k]*X.entries[k*X.numCols + j];
                }

                X.entries[xIndex] = (B.entries[xIndex] - sum) / diag;
            }
        }

        return X;
    }


    /**
     * Solves the linear system of equations given by {@code U*X=I} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix and {@code I} is the {@link Matrix#isI() identity}
     * matrix of appropriate size.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @return The solution to {@code X} in the linear system {@code U*X=B}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    public Matrix solveIdentity(Matrix U) {
        double sum, diag;
        int uIndex, xIndex;
        int n = U.numRows;
        Matrix X = new Matrix(U.shape);

        X.entries[X.entries.length-1] = 1.0/U.entries[n*n-1];

        for(int j=0; j<U.numCols; j++) {
            for(int i=n-2; i>-1; i--) {
                sum = (i == j) ? 1 : 0;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                uIndex += i+1;
                diag = U.entries[i*(n+1)];

                if(diag==0) throw new SingularMatrixException("Cannot solve linear system.");

                for(int k=i+1; k<n; k++) {
                    sum -= U.entries[uIndex++]*X.entries[k*X.numCols + j];
                }

                X.entries[xIndex] = sum / diag;
            }
        }

        return X;
    }


    /**
     * Solves a special case of the linear system {@code U*X=L} for {@code X} where the coefficient matrix {@code U}
     * is an {@link Matrix#isTriU() upper triangular} matrix and the constant matrix {@code L} is
     * {@link Matrix#isTriL() lower triangular}.
     *
     * @param U Upper triangular coefficient matrix
     * @param L Lower triangular constant matrix.
     * @return The result of solving the linear system {@code U*X=L} for the matrix {@code X}.
     */
    public Matrix solveLower(Matrix U, Matrix L) {
        double sum, diag;
        int uIndex, xIndex;
        int n = L.numRows;
        double uValue = U.entries[U.entries.length-1];
        int rowOffset = (n-1)*n;
        Matrix X = new Matrix(L.shape);

        for(int j=0; j<n; j++) {
            X.entries[rowOffset] = L.entries[rowOffset++]/uValue;

            for(int i=L.numCols-2; i>=0; i--) {
                sum = 0;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                diag = U.entries[i*(n+1)];

                if(diag==0) throw new SingularMatrixException("Cannot solve linear system.");

                for(int k=i+1; k<n; k++) {
                    sum += U.entries[uIndex + k]*X.entries[k*X.numCols + j];
                }

                X.entries[xIndex] = (L.entries[xIndex] - sum) / diag;
            }
        }

        return X;
    }
}
