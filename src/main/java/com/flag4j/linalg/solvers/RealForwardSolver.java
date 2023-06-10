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
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector.
 */
public class RealForwardSolver implements LinearSolver<Matrix, Vector> {

    /**
     * Flag which indicates if lower triangular matrix is unit lower triangular (i.e. ones along principle diagonal)
     * or not.
     * True indicates unit lower triangular and false indicates simply lower triangular.
     */
    protected boolean isUnit;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public RealForwardSolver() {
        super();
        isUnit = false;
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     *               - If true, the coefficient matrix is expected to be unit lower triangular. <br>
     *               - If true, the coefficient matrix is expected to be lower triangular.
     */
    public RealForwardSolver(boolean isUnit) {
        super();
        this.isUnit = isUnit;
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and a vector {@code b}.
     * That is, solves the linear system {@code L*x=b} where {@code L} is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param b Constant vector.
     * @return The result of solving the linear system {@code L*x=b} where {@code L} is a lower triangular matrix.
     * @throws SingularMatrixException If
     */
    @Override
    public Vector solve(Matrix L, Vector b) {
        return isUnit ? solveUnitLower(L, b) : solveLower(L, b);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and a matrix {@code B}.
     * That is, solves the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param B Constant Matrix.
     * @return The result of solving the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     *      * the principle diagonal).
     */
    @Override
    public Matrix solve(Matrix L, Matrix B) {
        return isUnit ? solveUnitLower(L, B) : solveLower(L, B);
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b Vector of constants in the linear system.
     * @return The solution of {@code x} for the linear system {@code L*x=b}.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    private Vector solveUnitLower(Matrix L, Vector b) {
        double sum;
        int lIndexStart;
        Vector x = new Vector(L.numRows);

        x.entries[0] = b.entries[0];

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--) {
                sum += L.entries[lIndexStart + j]*x.entries[j];
            }

            x.entries[i] = b.entries[i]-sum;
        }

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b Vector of constants in the linear system.
     * @return The solution of {@code x} for the linear system {@code L*x=b}.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private Vector solveLower(Matrix L, Vector b) {
        double sum, diag;
        int lIndexStart;
        Vector x = new Vector(L.numRows);

        x.entries[0] = b.entries[0];

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            diag = L.entries[i*(L.numCols + 1)];
            if(diag==0) {
                throw new SingularMatrixException("Cannot solve linear system.");
            }

            for(int j=i-1; j>-1; j--) {
                sum += L.entries[lIndexStart + j]*x.entries[j];
            }

            x.entries[i] = (b.entries[i]-sum)/diag;
        }

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of {@code X} for the linear system {@code L*X=b}.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    private Matrix solveUnitLower(Matrix L, Matrix B) {
        double sum;
        int lIndexStart, xIndex;
        Matrix X = new Matrix(B.shape);

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j];

            for(int i=1; i<L.numRows; i++) {
                sum = 0;
                lIndexStart = i*(L.numCols + 1) - 1;
                xIndex = i*X.numCols + j;

                for(int k=i-1; k>-1; k--) {
                    sum += L.entries[lIndexStart--]*X.entries[k*X.numCols + j];
                }

                X.entries[xIndex] = B.entries[xIndex] - sum;
            }
        }

        return X;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of {@code X} for the linear system {@code L*X=b}.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private Matrix solveLower(Matrix L, Matrix B) {
        double sum, diag;
        int lIndexStart, xIndex;
        Matrix X = new Matrix(B.shape);

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j];

            for(int i=1; i<L.numRows; i++) {
                sum = 0;
                lIndexStart = i*(L.numCols + 1) - 1;
                xIndex = i*X.numCols + j;

                diag = L.entries[i*(L.numCols + 1)];
                if(diag==0) {
                    throw new SingularMatrixException("Cannot solve linear system.");
                }

                for(int k=i-1; k>-1; k--) {
                    sum += L.entries[lIndexStart--]*X.entries[k*X.numCols + j];
                }

                X.entries[xIndex] = (B.entries[xIndex] - sum) / diag;
            }
        }

        return X;
    }
}
