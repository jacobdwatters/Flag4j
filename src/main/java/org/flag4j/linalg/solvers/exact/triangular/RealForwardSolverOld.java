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

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.SingularMatrixException;


/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector.
 */
public class RealForwardSolverOld extends ForwardSolverOld<MatrixOld, VectorOld, double[]> {

    /**
     * For computing determinant of lower triangular matrix during solve.
     */
    private double det;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public RealForwardSolverOld() {
        super(false, false);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     *               - If true, the coefficient matrix is expected to be unit lower triangular. <br>
     *               - If true, the coefficient matrix is expected to be lower triangular.
     */
    public RealForwardSolverOld(boolean isUnit) {
        super(isUnit, false);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     *               - If true, the coefficient matrix is expected to be unit lower triangular. <br>
     *               - If true, the coefficient matrix is expected to be lower triangular.
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    public RealForwardSolverOld(boolean isUnit, boolean enforceLower) {
        super(isUnit, enforceLower);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public double getDet() {
        return det;
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
    public VectorOld solve(MatrixOld L, VectorOld b) {
        ParameterChecks.ensureSquareMatrix(L.shape);
        ParameterChecks.ensureEquals(L.numRows, b.size);
        return isUnit ? solveUnitLower(L, b) : solveLower(L, b);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and a matrix {@code B}.
     * That is, solves the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param B Constant MatrixOld.
     * @return The result of solving the linear system {@code L*X=B} where {@code L} is a lower triangular matrix.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     *      * the principle diagonal).
     */
    @Override
    public MatrixOld solve(MatrixOld L, MatrixOld B) {
        ParameterChecks.ensureSquareMatrix(L.shape);
        ParameterChecks.ensureEquals(L.numRows, B.numRows);
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
    public MatrixOld solveIdentity(MatrixOld L) {
        ParameterChecks.ensureSquareMatrix(L.shape);
        return isUnit ? solveUnitLowerIdentity(L) : solveLowerIdentity(L);
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b VectorOld of constants in the linear system.
     * @return The solution of {@code x} for the linear system {@code L*x=b}.
     */
    private VectorOld solveUnitLower(MatrixOld L, VectorOld b) {
        checkParams(L, b.size);

        double sum;
        int lIndexStart;
        x = new VectorOld(L.numRows);
        det = L.numRows; // Since it is unit lower, matrix has full rank.

        x.entries[0] = b.entries[0];

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--) {
                sum += L.entries[lIndexStart + j]*x.entries[j];
            }

            x.entries[i] = b.entries[i]-sum;
        }

        // No need to check if matrix is singular since it has full rank.
        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b VectorOld of constants in the linear system.
     * @return The solution of {@code x} for the linear system {@code L*x=b}.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private VectorOld solveLower(MatrixOld L, VectorOld b) {
        checkParams(L, b.size);

        double sum, diag;
        int lIndexStart;
        x = new VectorOld(L.numRows);
        det = L.entries[0];
        x.entries[0] = b.entries[0]/det;

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            diag = L.entries[i*(L.numCols + 1)];
            det *= diag;

            for(int j=i-1; j>-1; j--) {
                sum += L.entries[lIndexStart + j]*x.entries[j];
            }

            x.entries[i] = (b.entries[i]-sum)/diag;
        }

        checkSingular(Math.abs(det), L.numRows, L.numCols); // Ensure matrix is not singular.

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B MatrixOld of constants in the linear system.
     * @return The solution of {@code X} for the linear system {@code L*X=b}.
     */
    private MatrixOld solveUnitLower(MatrixOld L, MatrixOld B) {
        checkParams(L, B.numRows);

        double sum;
        int lIndexStart, xIndex;
        X = new MatrixOld(B.shape);
        xCol = new double[L.numRows];
        det = L.numRows; // Since it is unit lower, matrix has full rank.

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j];

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = 0;
                lIndexStart = i*(L.numCols + 1) - 1;
                xIndex = i*X.numCols + j;

                for(int k=i-1; k>-1; k--) {
                    sum += L.entries[lIndexStart--]*X.entries[k*X.numCols + j];
                }


                xCol[i] = X.entries[xIndex] = B.entries[xIndex] - sum;
            }
        }

        // No need to check if matrix is singular since it has full rank.
        return X;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B MatrixOld of constants in the linear system.
     * @return The solution of {@code X} for the linear system {@code L*X=b}.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private MatrixOld solveLower(MatrixOld L, MatrixOld B) {
        checkParams(L, B.numRows);

        double sum;
        double diag;
        int lIndexStart, xIndex;
        X = new MatrixOld(B.shape);
        xCol = new double[L.numRows];
        det = L.entries[0];

        for(int j=0; j<B.numCols; j++) {
            X.entries[j] = B.entries[j]/L.entries[0];

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = 0;
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;

                diag = L.entries[i*(L.numCols + 1)];

                if(j == 0) det *= diag;

                for(int k=0; k<i; k++) {
                    sum += L.entries[lIndexStart++]*xCol[k];
                }

                double value = (B.entries[xIndex] - sum) / diag;
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular and the constant matrix
     * is the identity matrix.
     * @param L Unit lower triangular matrix.
     * @return The solution of {@code X} for the linear system {@code L*X=I}.
     */
    private MatrixOld solveUnitLowerIdentity(MatrixOld L) {
        checkParams(L, L.numRows);

        double sum;
        int lIndexStart, xIndex;
        X = new MatrixOld(L.shape);
        xCol = new double[L.numRows];
        det = L.numRows; // Since it is unit lower, matrix has full rank.

        X.entries[0] = 1.0;

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? 1.0 : 0.0;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;

                for(int k=0; k<i; k++) {
                    sum -= L.entries[lIndexStart++]*xCol[k];
                }

                xCol[i] = X.entries[xIndex] = sum;
            }
        }

        // No need to check if matrix is singular since it has full rank.
        return X;
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
    private MatrixOld solveLowerIdentity(MatrixOld L) {
        checkParams(L, L.numRows);

        double sum, diag;
        int lIndexStart, xIndex;
        X = new MatrixOld(L.shape);
        xCol = new double[L.numRows];
        det = L.entries[0];
        X.entries[0] = 1.0/det;

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? 1.0 : 0.0;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;
                diag = L.entries[i*(L.numCols + 1)];

                if(j==0) det*=diag;

                for(int k=0; k<i; k++) {
                    sum -= L.entries[lIndexStart++]*xCol[k];
                }

                double value = sum / diag;
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }
}
