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

package org.flag4j.linalg.solvers.exact.triangular;


import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular complex dense matrix
 * and the constant vector/matrix is complex and dense.
 */
public class ComplexForwardSolver extends ForwardSolver<CMatrix, CVector, Complex128[]> {


    /**
     * For computing determinant of lower triangular matrix during solve.
     */
    private Complex128 det;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public ComplexForwardSolver() {
        super(false, false);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not.
     * <ul>
     *     <li>If {@code true}, the coefficient matrix is expected to be unit lower triangular.</li>
     *     <li>If {@code false}, the coefficient matrix is expected to be lower triangular.</li>
     * </ul>
     */
    public ComplexForwardSolver(boolean isUnit) {
        super(isUnit, false);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not. <br>
     * <ul>
     *     <li>If {@code true}, the coefficient matrix is expected to be unit lower triangular.</li>
     *     <li>If {@code false}, the coefficient matrix is expected to be lower triangular.</li>
     * </ul>
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    public ComplexForwardSolver(boolean isUnit, boolean enforceLower) {
        super(isUnit, enforceLower);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public Complex128 getDet() {
        return det;
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix L and a vector b.
     * That is, solves the linear system L*x=b where L is a lower triangular matrix.
     * @param L Unit lower triangular coefficient matrix. If {@code L} is not unit lower triangular, it will be treated
     *          as if it were.
     * @param b Constant vector.
     * @return The result of solving the linear system L*x=b where L is a lower triangular matrix.
     */
    @Override
    public CVector solve(CMatrix L, CVector b) {
        ValidateParameters.ensureSquareMatrix(L.shape);
        ValidateParameters.ensureEquals(L.numRows, b.size);
        return isUnit ? solveUnitLower(L, b) : solveLower(L, b);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix L and a matrix B.
     * That is, solves the linear system L*X=B where L is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param B Constant Matrix.
     * @return The result of solving the linear system L*X=B where L is a lower triangular matrix.
     * If the lower triangular matrix L is singular (i.e. has a zero on the principle diagonal).
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    @Override
    public CMatrix solve(CMatrix L, CMatrix B) {
        ValidateParameters.ensureSquareMatrix(L.shape);
        ValidateParameters.ensureEquals(L.numRows, B.numRows);
        return isUnit ? solveUnitLower(L, B) : solveLower(L, B);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix L and the identity matrix.
     * That is, solves the linear system L*X=I where L is a lower triangular matrix and I is
     * the appropriately sized identity matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @return The result of solving the linear system L*X=B where L is a lower triangular matrix.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    public CMatrix solveIdentity(CMatrix L) {
        ValidateParameters.ensureSquareMatrix(L.shape);
        return isUnit ? solveUnitLowerIdentity(L) : solveLowerIdentity(L);
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution of x for the linear system L*x=b.
     */
    private CVector solveUnitLower(CMatrix L, CVector b) {
        checkParams(L, b.size);

        Complex128 sum;
        int lIndexStart;
        CVector x = new CVector(L.numRows);
        det = new Complex128(L.numRows); // Since it is unit lower, matrix has full rank.

        x.data[0] = b.data[0];

        for(int i=1; i<L.numRows; i++) {
            sum = Complex128.ZERO;
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--) {
                sum = sum.add(L.data[lIndexStart + j].mult((Complex128) x.data[j]));
            }

            x.data[i] = b.data[i].sub(sum);
        }

        return x; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular and the constant matrix
     * is the identity matrix.
     * @param L Unit lower triangular matrix.
     * @return The solution of X for the linear system L*X=I.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     * the principle diagonal).
     */
    private CMatrix solveUnitLowerIdentity(CMatrix L) {
        checkParams(L, L.numRows);

        Complex128 sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(L.shape);
        det = new Complex128(L.numRows); // Since it is unit lower, matrix has full rank.
        xCol = new Complex128[L.numRows];

        X.data[0] = Complex128.ONE;

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? Complex128.ONE : Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;

                for(int k=0; k<i; k++) {
                    sum = sum.sub(L.data[lIndexStart++].mult((Complex128) xCol[k]));
                }

                X.data[xIndex] = sum;
                xCol[i] = sum;
            }
        }

        return X; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system L*X=I where the coefficient matrix L is lower triangular and the
     * constant matrix I is the appropriately sized identity matrix.
     * @param L Unit lower triangular matrix (Note, this is not checked).
     *          If {@code L} is not lower triangular, it will be treated as if it were. No error will be thrown.
     * @return The solution of X for the linear system L*X=I.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private CMatrix solveLowerIdentity(CMatrix L) {
        checkParams(L, L.numRows);

        Complex128 sum, diag;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(L.shape);
        det = (Complex128) L.data[0];
        xCol = new Complex128[L.numRows];

        X.data[0] = L.data[0].multInv();

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                if(j==0) det = det.mult((Complex128) L.data[i*L.numCols + i]);

                sum = (i==j) ? Complex128.ONE : Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;
                diag = (Complex128) L.data[i*(L.numCols + 1)];

                for(int k=0; k<i; k++)
                    sum = sum.sub(L.data[lIndexStart++].mult((Complex128) xCol[k]));

                Complex128 value = sum.div(diag);
                X.data[xIndex] = value;
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
     * @return The solution of x for the linear system L*x=b.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private CVector solveLower(CMatrix L, CVector b) {
        checkParams(L, b.size);

        Complex128 sum, diag;
        int lIndexStart;
        CVector x = new CVector(L.numRows);
        det = (Complex128) L.data[0];
        x.data[0] = b.data[0].div((Complex128) L.data[0]);

        for(int i=1; i<L.numRows; i++) {
            sum = Complex128.ZERO;
            lIndexStart = i*L.numCols;
            diag = (Complex128) L.data[i*(L.numCols + 1)];
            det = det.mult(diag);

            for(int j=i-1; j>-1; j--) {
                sum = sum.add(L.data[lIndexStart + j].mult((Complex128) x.data[j]));
            }

            x.data[i] = b.data[i].sub(sum).div(diag);
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of X for the linear system L*X=b.
     */
    private CMatrix solveUnitLower(CMatrix L, CMatrix B) {
        checkParams(L, B.numRows);

        Complex128 sum;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);
        det = new Complex128(L.numRows); // Since it is unit lower, matrix has full rank.
        xCol = new Complex128[L.numRows];

        for(int j=0; j<B.numCols; j++) {
            X.data[j] = B.data[j];

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;

                for(int k=0; k<i; k++) {
                    sum = sum.add(L.data[lIndexStart++].mult((Complex128) xCol[k]));
                }

                Complex128 value = B.data[xIndex].sub(sum);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        return X; // No need to check singularity since the matrix is full rank.
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of X for the linear system L*X=b.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private CMatrix solveLower(CMatrix L, CMatrix B) {
        checkParams(L, B.numRows);

        Complex128 sum;
        Complex128 diag;
        int lIndexStart, xIndex;
        CMatrix X = new CMatrix(B.shape);
        det = (Complex128) L.data[0];
        xCol = new Complex128[L.numRows];

        for(int j=0; j<B.numCols; j++) {
            X.data[j] = B.data[j].div((Complex128) L.data[0]);

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=1; i<L.numRows; i++) {
                sum = Complex128.ZERO;
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;
                diag = (Complex128) L.data[i*(L.numCols + 1)];

                if(j==0) det = det.mult(diag);

                for(int k=0; k<i; k++) {
                    sum = sum.add(L.data[lIndexStart++].mult((Complex128) xCol[k]));
                }

                Complex128 value = B.data[xIndex].sub(sum).div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }
}
