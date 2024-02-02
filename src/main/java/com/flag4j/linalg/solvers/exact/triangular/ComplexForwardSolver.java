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

package com.flag4j.linalg.solvers.exact.triangular;


import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.util.ParameterChecks;


/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular complex dense matrix
 * and the constant vector is a complex dense vector.
 */
public class ComplexForwardSolver extends ForwardSolver<CMatrix, CVector, CNumber[]> {


    /**
     * For computing determinant of lower triangular matrix during solve.
     */
    private CNumber det;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public  ComplexForwardSolver() {
        super(false, false);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     *               - If true, the coefficient matrix is expected to be unit lower triangular. <br>
     *               - If true, the coefficient matrix is expected to be lower triangular.
     */
    public ComplexForwardSolver(boolean isUnit) {
        super(isUnit, false);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     *               - If true, the coefficient matrix is expected to be unit lower triangular. <br>
     *               - If true, the coefficient matrix is expected to be lower triangular.
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    public ComplexForwardSolver(boolean isUnit, boolean enforceLower) {
        super(isUnit, enforceLower);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public CNumber getDet() {
        return det;
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
        ParameterChecks.assertSquareMatrix(L.shape);
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
        ParameterChecks.assertSquareMatrix(L.shape);
        ParameterChecks.assertEquals(L.numRows, B.numRows);
        return isUnit ? solveUnitLower(L, B) : solveLower(L, B);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and the identity matrix.
     * That is, solves the linear system {@code L*X=I} where {@code L} is a lower triangular matrix and {@code I} is
     * the appropriately sized identity matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @return The result of solving the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    public CMatrix solveIdentity(CMatrix L) {
        ParameterChecks.assertSquareMatrix(L.shape);
        return isUnit ? solveUnitLowerIdentity(L) : solveLowerIdentity(L);
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution of {@code x} for the linear system {@code L*x=b}.
     */
    private CVector solveUnitLower(CMatrix L, CVector b) {
        checkParams(L, b.size);

        CNumber sum;
        int lIndexStart;
        CVector x = new CVector(L.numRows);
        det = new CNumber(L.numRows); // Since it is unit lower, matrix has full rank.

        x.entries[0] = b.entries[0].copy();

        for(int i=1; i<L.numRows; i++) {
            sum = new CNumber();
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--) {
                sum.addEq(L.entries[lIndexStart + j].mult(x.entries[j]));
            }

            x.entries[i] = b.entries[i].sub(sum);
        }

        return x; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular and the constant matrix
     * is the identity matrix.
     * @param L Unit lower triangular matrix.
     * @return The solution of {@code X} for the linear system {@code L*X=I}.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    private CMatrix solveUnitLowerIdentity(CMatrix L) {
        checkParams(L, L.numRows);

        CNumber sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(L.shape);
        det = new CNumber(L.numRows); // Since it is unit lower, matrix has full rank.
        xCol = new CNumber[L.numRows];

        X.entries[0] = CNumber.one();

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? CNumber.one() : CNumber.zero();
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;

                for(int k=0; k<i; k++) {
                    sum.subEq(L.entries[lIndexStart++].mult(xCol[k]));
                }

                X.entries[xIndex] = sum;
                xCol[i] = sum;
            }
        }

        return X; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system {@code L*X=I} where the coefficient matrix {@code L} is lower triangular and the
     * constant matrix {@code I} is the appropriately sized identity matrix.
     * @param L Unit lower triangular matrix (Note, this is not checked).
     *          If {@code L} is not lower triangular, it will be treated as if it were. No error will be thrown.
     * @return The solution of {@code X} for the linear system {@code L*X=I}.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private CMatrix solveLowerIdentity(CMatrix L) {
        checkParams(L, L.numRows);

        CNumber sum, diag;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(L.shape);
        det = L.entries[0].copy();
        xCol = new CNumber[L.numRows];

        X.entries[0] = L.entries[0].multInv();

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? CNumber.one() : CNumber.zero();
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;
                diag = L.entries[i*(L.numCols + 1)];
                det.multEq(diag);

                for(int k=0; k<i; k++) {
                    sum.subEq(L.entries[lIndexStart++].mult(xCol[k]));
                }

                CNumber value = sum.div(diag);
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
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
        checkParams(L, b.size);

        CNumber sum, diag;
        int lIndexStart;
        CVector x = new CVector(L.numRows);
        det = L.entries[0].copy();
        x.entries[0] = b.entries[0].div(L.entries[0]);

        for(int i=1; i<L.numRows; i++) {
            sum = new CNumber();
            lIndexStart = i*L.numCols;
            diag = L.entries[i*(L.numCols + 1)];
            det.multEq(diag);

            for(int j=i-1; j>-1; j--) {
                sum.addEq(L.entries[lIndexStart + j].mult(x.entries[j]));
            }

            x.entries[i] = b.entries[i].sub(sum).div(diag);
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of {@code X} for the linear system {@code L*X=b}.
     */
    private CMatrix solveUnitLower(CMatrix L, CMatrix B) {
        checkParams(L, B.numRows);

        CNumber sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);
        det = new CNumber(L.numRows); // Since it is unit lower, matrix has full rank.
        xCol = new CNumber[L.numRows];

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j].copy();

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = new CNumber();
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;

                for(int k=0; k<i; k++) {
                    sum.addEq(L.entries[lIndexStart++].mult(xCol[k]));
                }

                CNumber value = B.entries[xIndex].sub(sum);
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        return X; // No need to check singularity since the matrix is full rank.
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
        checkParams(L, B.numRows);

        CNumber sum;
        CNumber diag;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);
        det = L.entries[0].copy();
        xCol = new CNumber[L.numRows];

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j].div(L.entries[0]);

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = CNumber.zero();
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;
                diag = L.entries[i*(L.numCols + 1)];
                det.multEq(diag);

                for(int k=0; k<i; k++) {
                    sum.addEq(L.entries[lIndexStart++].mult(xCol[k]));
                }

                CNumber value = B.entries[xIndex].sub(sum).div(diag);
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }
}
