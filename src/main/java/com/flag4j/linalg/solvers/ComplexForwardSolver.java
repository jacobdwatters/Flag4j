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


import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.util.ParameterChecks;


/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular complex dense matrix
 * and the constant vector is a complex dense vector.
 */
public class ComplexForwardSolver implements LinearSolver<CMatrix, CVector> {


    /**
     * Flag which indicates if lower triangular matrix is unit lower triangular (i.e. ones along principle diagonal)
     * or not.
     * True indicates unit lower triangular and false indicates simply lower triangular.
     */
    protected boolean isUnit;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public ComplexForwardSolver() {
        super();
        isUnit = false;
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     *               - If true, the coefficient matrix is expected to be unit lower triangular. <br>
     *               - If true, the coefficient matrix is expected to be lower triangular.
     */
    public ComplexForwardSolver(boolean isUnit) {
        super();
        this.isUnit = isUnit;
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and a vector {@code b}.
     * That is, solves the linear system {@code L*x=b} where {@code L} is a lower triangular matrix.
     * @param L Unit lower triangular coefficient matrix. If {@code L} is not unit lower triangular, it will be treated
     *          as if it were.
     * @param b Constant vector.
     * @return The result of solving the linear system {@code L*x=b} where {@code L} is a lower triangular matrix.
     */
    @Override
    public CVector solve(CMatrix L, CVector b) {
        ParameterChecks.assertSquare(L.shape);
        ParameterChecks.assertEquals(L.numRows, b.size);
        return isUnit ? solveUnitLower(L, b) : solveLower(L, b);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and a matrix {@code B}.
     * That is, solves the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param B Constant Matrix.
     * @return The result of solving the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * If the lower triangular matrix {@code L} is singular (i.e. has a zero on the principle diagonal).
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    @Override
    public CMatrix solve(CMatrix L, CMatrix B) {
        ParameterChecks.assertSquare(L.shape);
        ParameterChecks.assertEquals(L.numRows, B.numRows);
        return isUnit ? solveUnitLower(L, B) : solveLower(L, B);
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution of {@code x} for the linear system {@code L*x=b}.
     */
    private CVector solveUnitLower(CMatrix L, CVector b) {
        CNumber sum;
        int lIndexStart;
        CVector x = new CVector(L.numRows);

        x.entries[0] = b.entries[0].copy();

        for(int i=1; i<L.numRows; i++) {
            sum = new CNumber();
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--) {
                sum.addEq(L.entries[lIndexStart + j].mult(x.entries[j]));
            }

            x.entries[i] = b.entries[i].sub(sum);
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
    private CVector solveLower(CMatrix L, CVector b) {
        CNumber sum, diag;
        int lIndexStart;
        CVector x = new CVector(L.numRows);

        if(L.entries[0].equals(CNumber.ZERO)) {
            throw new SingularMatrixException("Cannot solve linear system.");
        }

        x.entries[0] = b.entries[0].div(L.entries[0]);

        for(int i=1; i<L.numRows; i++) {
            sum = new CNumber();
            lIndexStart = i*L.numCols;

            diag = L.entries[i*(L.numCols + 1)];
            if(diag.equals(CNumber.ZERO)) {
                throw new SingularMatrixException("Cannot solve linear system.");
            }

            for(int j=i-1; j>-1; j--) {
                sum.addEq(L.entries[lIndexStart + j].mult(x.entries[j]));
            }

            x.entries[i] = b.entries[i].sub(sum).div(diag);
        }

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of {@code X} for the linear system {@code L*X=b}.
     */
    private CMatrix solveUnitLower(CMatrix L, CMatrix B) {
        CNumber sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j].copy();

            for(int i=1; i<L.numRows; i++) {
                sum = new CNumber();
                lIndexStart = i*(L.numCols + 1) - 1;
                xIndex = i*X.numCols + j;

                for(int k=i-1; k>-1; k--) {
                    sum.addEq(L.entries[lIndexStart--].mult(X.entries[k*X.numCols + j]));
                }

                X.entries[xIndex] = B.entries[xIndex].sub(sum);
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
    private CMatrix solveLower(CMatrix L, CMatrix B) {
        CNumber sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);

        // Only check the diagonal has no zeros once.
        if(zeroOnDiag(L)) {
            throw new SingularMatrixException("Cannot solve linear system.");
        }

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j].div(L.entries[0]);

            for(int i=1; i<L.numRows; i++) {
                sum = new CNumber();
                lIndexStart = i*(L.numCols + 1) - 1;
                xIndex = i*X.numCols + j;

                for(int k=i-1; k>-1; k--) {
                    sum.addEq(L.entries[lIndexStart--].mult(X.entries[k*X.numCols + j]));
                }

                X.entries[xIndex] = B.entries[xIndex].sub(sum).div(L.entries[i*(L.numCols + 1)]);
            }
        }

        return X;
    }


    /**
     * Checks if a matrix has a zero on the diagonal of a matrix.
     * @param src Matrix of interest. Assumed to be square.
     * @return True if the matrix has a zero on the diagonal. False otherwise.
     */
    private boolean zeroOnDiag(CMatrix src) {
        boolean result = false;

        for(int i=0; i<src.numRows; i++) {
            if(src.entries[i*(src.numCols  + 1)].equals(CNumber.ZERO)) {
                result = true;
                break; // No need to continue.
            }
        }

        return result;
    }
}
