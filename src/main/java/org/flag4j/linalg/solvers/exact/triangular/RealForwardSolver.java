/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector.
 */
public class RealForwardSolver extends ForwardSolver<Matrix, Vector, double[]> {

    // TODO: In several implementations, a column is temporarily copied. This is likely only worth it for matrices
    //  larger than a few entries. If the matrix is small consider not doing this. Also should investigate the minimum size
    //  it is likely "worth it".

    /**
     * For computing determinant of lower triangular matrix during solve.
     */
    private double det;


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular.
     */
    public RealForwardSolver() {
        super(false, true);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not.
     * <ul>
     *     <li>If {@code true}, the coefficient matrix is expected to be unit lower triangular.</li>
     *     <li>If {@code false}, the coefficient matrix is expected to be lower triangular but not necessarily unit lower.</li>
     * </ul>
     */
    public RealForwardSolver(boolean isUnit) {
        super(isUnit, true);
    }


    /**
     * Creates a solver to solve a linear system where the coefficient matrix is lower triangular or unit lower triangular.
     * @param isUnit Flag which indicates if the coefficient matrix is unit lower triangular or not.
     * <ul>
     *     <li>If {@code true}, the coefficient matrix is expected to be unit lower triangular.</li>
     *     <li>If {@code false}, the coefficient matrix is expected to be lower triangular but not necessarily unit lower.</li>
     * </ul>
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    public RealForwardSolver(boolean isUnit, boolean enforceLower) {
        super(isUnit, enforceLower);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public double getDet() {
        return det;
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix L and a vector b.
     * That is, solves the linear system L*x=b where L is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param b Constant vector.
     * @return The result of solving the linear system L*x=b where L is a lower triangular matrix.
     * @throws SingularMatrixException If
     */
    @Override
    public Vector solve(Matrix L, Vector b) {
        return isUnit ? solveUnitLower(L, b) : solveLower(L, b);
    }


    /**
     * Performs forward substitution for a unit lower triangular matrix L and a matrix B.
     * That is, solves the linear system L*X=B where L is a lower triangular matrix.
     * @param L Lower triangular coefficient matrix. If {@code L} is not lower triangular, it will be treated
     *          as if it were.
     * @param B Constant Matrix.
     * @return The result of solving the linear system L*X=B where L is a lower triangular matrix.
     * @throws SingularMatrixException If the matrix lower triangular {@code L} is singular (i.e. has a zero on
     *      * the principle diagonal).
     */
    @Override
    public Matrix solve(Matrix L, Matrix B) {
        return isUnit ? solveUnitLower(L, B) : solveLower(L, B);
    }


    /**
     * Solves a linear system <b>L*X=P</b> for <b>X</b> where <b>L</b> is a lower triangular matrix and
     * <b>P</b> is a permutation matrix.
     * @param L Lower triangular coefficient matrix <b>L</b>.
     * @param P Constant permutation matrix <b>P</b>.
     * @return The solution to <b>X</b> in the linear system <b>L*X=P</b>.
     */
    @Override
    public Matrix solve(Matrix L, PermutationMatrix P) {
        return isUnit ? solveUnitPerm(L, P) : solvePerm(L, P);
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
    public Matrix solveIdentity(Matrix L) {
        ValidateParameters.ensureSquareMatrix(L.shape);
        return isUnit ? solveUnitLowerIdentity(L) : solveLowerIdentity(L);
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b Vector of constants in the linear system.
     * @return The solution of x for the linear system L*x=b.
     */
    private Vector solveUnitLower(Matrix L, Vector b) {
        checkParams(L, b.size);

        double sum;
        int lIndexStart;
        x = new Vector(L.numRows);
        det = L.numRows; // Since it is unit lower, matrix has full rank.

        x.data[0] = b.data[0];

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--)
                sum += L.data[lIndexStart + j]*x.data[j];

            x.data[i] = b.data[i]-sum;
        }

        // No need to check if matrix is singular since it has full rank.
        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param b Vector of constants in the linear system.
     * @return The solution of x for the linear system L*x=b.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private Vector solveLower(Matrix L, Vector b) {
        checkParams(L, b.size);

        double sum, diag;
        int lIndexStart;
        x = new Vector(L.numRows);
        det = L.data[0];
        x.data[0] = b.data[0]/det;

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            diag = L.data[i*(L.numCols + 1)];
            det *= diag;

            for(int j=i-1; j>-1; j--)
                sum += L.data[lIndexStart + j]*x.data[j];

            x.data[i] = (b.data[i]-sum)/diag;
        }

        checkSingular(Math.abs(det), L.numRows, L.numCols); // Ensure matrix is not singular.

        return x;
    }


    /**
     * Solves a linear system where the coefficient matrix is unit lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of X for the linear system L*X=b.
     */
    private Matrix solveUnitLower(Matrix L, Matrix B) {
        checkParams(L, B.numRows);

        double sum;
        int lIndexStart, xIndex;
        X = new Matrix(B.shape);
        xCol = new double[L.numRows];
        det = L.numRows; // Since it is unit lower, matrix has full rank.

        for(int j=0; j<B.numCols; j++) {
            X.data[j] = B.data[j];

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = 0;
                lIndexStart = i*(L.numCols + 1) - 1;
                xIndex = i*X.numCols + j;

                for(int k=i-1; k>-1; k--)
                    sum += L.data[lIndexStart--]*xCol[k];

                xCol[i] = X.data[xIndex] = B.data[xIndex] - sum;
            }
        }

        // No need to check if matrix is singular since it has full rank.
        return X;
    }


    /**
     * Solves a linear system where the coefficient matrix is lower triangular.
     * @param L Unit lower triangular matrix.
     * @param B Matrix of constants in the linear system.
     * @return The solution of X for the linear system L*X=b.
     * @throws SingularMatrixException If the lower triangular matrix {@code L} is singular (i.e. has a zero on the
     * principle diagonal).
     */
    private Matrix solveLower(Matrix L, Matrix B) {
        checkParams(L, B.numRows);

        double sum;
        double diag;
        int lIndexStart, xIndex;
        X = new Matrix(B.shape);
        xCol = new double[L.numRows];
        det = L.data[0];

        for(int j=0; j<B.numCols; j++) {
            X.data[j] = B.data[j]/L.data[0];

            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = 0;
                lIndexStart = i*L.numCols;
                xIndex = i*X.numCols + j;

                diag = L.data[i*(L.numCols + 1)];

                if(j == 0) det *= diag;

                for(int k=0; k<i; k++)
                    sum += L.data[lIndexStart++]*xCol[k];

                double value = (B.data[xIndex] - sum) / diag;
                X.data[xIndex] = value;
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
     * @return The solution of <b>X</b> for the linear system <b>L*X=I</b>.
     */
    private Matrix solveUnitLowerIdentity(Matrix L) {
        checkParams(L, L.numRows);

        double sum;
        int lIndexStart, xIndex;
        X = new Matrix(L.shape);
        xCol = new double[L.numRows];
        det = L.numRows; // Since it is unit lower, matrix has full rank.

        X.data[0] = 1.0;

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? 1.0 : 0.0;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;

                for(int k=0; k<i; k++)
                    sum -= L.data[lIndexStart++]*xCol[k];

                xCol[i] = X.data[xIndex] = sum;
            }
        }

        // No need to check if matrix is singular since it has full rank.
        return X;
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
    private Matrix solveLowerIdentity(Matrix L) {
        checkParams(L, L.numRows);

        double sum, diag;
        int lIndexStart, xIndex;
        X = new Matrix(L.shape);
        xCol = new double[L.numRows];
        det = L.data[0];
        X.data[0] = 1.0/det;

        for(int j=0; j<L.numCols; j++) {
            // Temporarily store column for better cache performance on innermost loop.
            for(int k=0; k<xCol.length; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=1; i<L.numRows; i++) {
                sum = (i==j) ? 1.0 : 0.0;
                lIndexStart = i*L.numCols;
                xIndex = lIndexStart + j;
                diag = L.data[i*(L.numCols + 1)];

                if(j==0) det*=diag;

                for(int k=0; k<i; k++)
                    sum -= L.data[lIndexStart++]*xCol[k];

                double value = sum / diag;
                X.data[xIndex] = xCol[i] = value;
            }
        }

        checkSingular(Math.abs(det), L.numRows, L.numCols); // Ensure matrix is not singular.

        return X;
    }


    /**
     * Solves a linear system <b>LX=P</b> where the coefficient matrix <b>L</b> is lower triangular and the
     * constant matrix <b>P</b> is a permutation matrix.
     * @param L Lower triangular coefficient matrix <b>L</b>.
     * @return The solution of <b>X</b> to the linear system <b>LX=P</b>.
     * @throws SingularMatrixException If {@code L} is singular (i.e. has a zero on the principle diagonal).
     */
    private Matrix solvePerm(Matrix L, PermutationMatrix P) {
        checkParams(L, P.size);

        final int n = P.size;
        int[] perm = P.getPermutation();
        X = new Matrix(P.size, P.size);
        det = L.data[0];
        xCol = new double[n];

        for (int j = 0; j < n; j++) {
            double bVal0 = (perm[0] == j) ? 1.0 : 0.0;
            X.data[j] = bVal0 / L.data[0];

            // Temporarily store column for better cache performance on innermost loop.
            for (int k = 0; k < n; k++)
                xCol[k] = X.data[k*n + j];

            for (int i = 1; i < n; i++) {
                double sum = 0.0;
                int lIndexStart = i*n;

                // Accumulate the dot product
                for (int k = 0; k < i; k++)
                    sum += L.data[lIndexStart + k]*xCol[k];

                double diag = L.data[i*(n + 1)];

                if (j == 0) det *= diag;

                double bVal = (perm[i] == j) ? 1.0 : 0.0;
                double value = (bVal - sum) / diag;

                X.data[lIndexStart + j] = xCol[i] = value;
            }
        }

        // If you want to check for singularity:
        checkSingular(Math.abs(det), n, n);

        return X;
    }


    /**
     * Solves a linear system <b>LX=P</b> where the coefficient matrix <b>L</b> is unit-lower triangular and the
     * constant matrix <b>P</b> is a permutation matrix.
     * @param L Unit lower triangular coefficient matrix <b>L</b>.
     * @return The solution of <b>X</b> to the linear system <b>LX=P</b>.
     */
    private Matrix solveUnitPerm(Matrix L, PermutationMatrix P) {
        checkParams(L, P.size);

        final int n = P.size;
        int[] perm = P.getPermutation();
        X = new Matrix(P.size, P.size);
        det = L.numRows;
        xCol = new double[n];

        for (int j = 0; j < n; j++) {
            double bVal0 = (perm[0] == j) ? 1.0 : 0.0;
            X.data[j] = bVal0;

            // Temporarily store column for better cache performance on innermost loop.
            for (int k = 0; k < n; k++)
                xCol[k] = X.data[k*n + j];

            for (int i = 1; i < n; i++) {
                double sum = 0.0;
                int lIndexStart = i*n;

                // Accumulate the dot product
                for (int k = 0; k < i; k++)
                    sum += L.data[lIndexStart + k]*xCol[k];

                double bVal = (perm[i] == j) ? 1.0 : 0.0;
                double value = bVal - sum;

                X.data[lIndexStart + j] = xCol[i] = value;
            }
        }

        // No need to check if matrix is singular since it has full rank.
        return X;
    }
}
