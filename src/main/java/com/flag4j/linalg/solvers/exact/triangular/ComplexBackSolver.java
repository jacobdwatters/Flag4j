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
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.SingularMatrixException;

/**
 * This solver solves linear systems of equations where the coefficient matrix in an upper triangular complex dense matrix
 * and the constant vector is a complex dense vector.
 */
public class ComplexBackSolver extends BackSolver<CMatrix, CVector, CNumber[]> {

    /**
     * For computing determinant of coefficient matrix during solve.
     */
    protected CNumber det;
    /**
     * For checking against other values.
     */
    private final CNumber z = CNumber.zero();


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices. Note, by default no check will
     * be made to ensure the coefficient matrix is upper triangular. If you would like to enforce this, see
     * {@link #ComplexBackSolver(boolean)}.
     */
    public ComplexBackSolver() {
        super(false);
    }


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public ComplexBackSolver(boolean enforceTriU) {
        super(enforceTriU);
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public CNumber getDet() {
        return det;
    }


    /**
     * Solves the linear system of equations given by {@code U*x=b} where the coefficient matrix {@code U}
     * is an upper triangular matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. if it has a zero along
     * the principle diagonal).
     */
    @Override
    public CVector solve(CMatrix U, CVector b) {
        checkParams(U, b.size);

        CNumber sum;
        int uIndex;
        int n = b.size;
        x = new CVector(U.numRows);
        det = U.entries[n*n-1];

        x.entries[n-1] = b.entries[n-1].div(det);

        for(int i=n-2; i>-1; i--) {
            sum = new CNumber();
            uIndex = i*U.numCols;

            CNumber diag = U.entries[i*(n+1)];
            det.multEq(diag);

            for(int j=i+1; j<n; j++) {
                sum.addEq(U.entries[uIndex + j].mult(x.entries[j]));
            }

            x.entries[i] = (b.entries[i].sub(sum)).div(diag);
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return x;
    }


    /**
     * Solves the linear system of equations given by {@code U*X=B} where the coefficient matrix {@code U}
     * is an upper triangular matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=B}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    @Override
    public CMatrix solve(CMatrix U, CMatrix B) {
        checkParams(U, B.numRows);

        CNumber sum, diag;
        int uIndex, xIndex;
        int n = B.numRows;
        X = new CMatrix(B.shape);
        det = U.entries[n*n-1].copy();

        xCol = new CNumber[n];

        for(int j=0; j<B.numCols; j++) {
            X.entries[(n-1)*X.numCols + j] = B.entries[(n-1)*X.numCols + j].div(U.entries[n*n-1]);

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.entries[k*X.numCols + j].copy();
            }

            for(int i=n-2; i>-1; i--) {
                sum = new CNumber();
                uIndex = i*U.numCols;
                xIndex = i*X.numCols + j;

                diag = U.entries[i*(n+1)];
                det.multEq(diag);

                for(int k=i+1; k<n; k++) {
                    sum.addEq(U.entries[uIndex + k].mult(xCol[k]));
                }

                CNumber value = B.entries[xIndex].sub(sum).div(diag);
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves the linear system of equations given by {@code U*X=I} where the coefficient matrix {@code U}
     * is an {@link CMatrix#isTriU() upper triangular} matrix and {@code I} is the {@link Matrix#isI() identity}
     * matrix of appropriate size.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @return The solution to {@code X} in the linear system {@code U*X=B}.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    public CMatrix solveIdentity(CMatrix U) {
        checkParams(U, U.numRows);

        CNumber sum, diag;
        int uIndex, xIndex;
        int n = U.numRows;
        X = new CMatrix(U.shape);
        det = U.entries[n*n-1].copy();

        xCol = new CNumber[n];
        X.entries[X.entries.length-1] = det.multInv();

        for(int j=0; j<U.numCols; j++) {
            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=n-2; i>-1; i--) {
                sum = (i == j) ? CNumber.one() : CNumber.zero();
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                uIndex += i+1;
                diag = U.entries[i*(n+1)];

                det.multEq(diag);

                for(int k=i+1; k<n; k++) {
                    sum.subEq(U.entries[uIndex++].mult(xCol[k]));
                }

                CNumber value = sum.div(diag);
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves a special case of the linear system {@code U*X=L} for {@code X} where the coefficient matrix {@code U}
     * is an {@link CMatrix#isTriU() upper triangular} matrix and the constant matrix {@code L} is
     * {@link CMatrix#isTriL() lower triangular}.
     *
     * @param U Upper triangular coefficient matrix
     * @param L Lower triangular constant matrix.
     * @return The result of solving the linear system {@code U*X=L} for the matrix {@code X}.
     */
    public CMatrix solveLower(CMatrix U, CMatrix L) {
        checkParams(U, L.numRows);

        CNumber sum, diag;
        int uIndex, xIndex;
        int n = L.numRows;
        CNumber uValue = U.entries[n*n-1];
        int rowOffset = (n-1)*n;
        X = new CMatrix(L.shape);
        det = uValue.copy();

        xCol = new CNumber[n];

        for(int j=0; j<n; j++) {
            X.entries[rowOffset] = L.entries[rowOffset++].div(uValue);

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.entries[k*X.numCols + j];
            }

            for(int i=L.numCols-2; i>=0; i--) {
                sum = CNumber.zero();
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                diag = U.entries[i*(n+1)];

                det.multEq(diag);

                for(int k=i+1; k<n; k++) {
                    sum.addEq(U.entries[uIndex + k].mult(xCol[k]));
                }

                CNumber value = L.entries[xIndex].sub(sum).div(diag);
                X.entries[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }
}
