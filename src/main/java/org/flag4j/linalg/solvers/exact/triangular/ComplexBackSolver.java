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


import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.util.exceptions.SingularMatrixException;


/**
 * This solver solves linear systems of equations where the coefficient matrix in an upper triangular complex dense matrix
 * and the constant vector is a complex dense vector.
 */
public class ComplexBackSolver extends BackSolver<CMatrix, CVector, Complex128[]> {
    
    /**
     * For computing determinant of coefficient matrix during solve.
     */
    protected Complex128 det;


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices. Note, by default no check will
     * be made to ensure the coefficient matrix is upper triangular. If you would like to enforce this, see
     * {@link #ComplexBackSolver(boolean)}.
     */
    public ComplexBackSolver() {
        super(true);
    }


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public ComplexBackSolver(boolean enforceTriU) {
        super(enforceTriU);
    }


    /**
     * Sets a flag indicating if an explicit check should be made that the coefficient matrix is singular.
     *
     * @param checkSingular Flag indicating if an explicit check should be made that the matrix is singular (or near singular).
     * <ul>
     *     <li>If {@code true}, an explicit singularity check will be made.</li>
     *     <li>If {@code false}, <em>no</em> check will be made.</li>
     * </ul>
     *
     * @return A reference to this back solver instance.
     */
    @Override
    public ComplexBackSolver setCheckSingular(boolean checkSingular) {
        super.setCheckSingular(checkSingular);
        return this;
    }


    /**
     * Gets the determinant computed during the last solve.
     */
    public Complex128 getDet() {
        return det;
    }


    /**
     * Solves the linear system of equations given by U*x=b where the coefficient matrix U
     * is an upper triangular matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @param b Vector of constants in the linear system.
     * @return The solution to x in the linear system A*x=b.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. if it has a zero along
     * the principle diagonal).
     */
    @Override
    public CVector solve(CMatrix U, CVector b) {
        checkParams(U, b.shape);

        Complex128 sum;
        int uIndex;
        int n = b.size;
        x = new CVector(U.numRows);
        det = U.data[n*n-1];

        x.data[n-1] = b.data[n-1].div(det);

        for(int i=n-2; i>-1; i--) {
            sum = Complex128.ZERO;
            uIndex = i*U.numCols;
            Complex128 diag = U.data[i*(n+1)];
            det = det.mult(diag);

            for(int j=i+1; j<n; j++)
                sum = sum.add(U.data[uIndex + j].mult(x.data[j]));

            x.data[i] = (b.data[i].sub(sum)).div(diag);
        }

        if(checkSingular) checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return x;
    }


    /**
     * Solves the linear system of equations given by U*X=B where the coefficient matrix U
     * is an upper triangular matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @param B Matrix of constants in the linear system.
     * @return The solution to X in the linear system A*X=B.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    @Override
    public CMatrix solve(CMatrix U, CMatrix B) {
        checkParams(U, B.shape);

        Complex128 sum, diag;
        int uIndex, xIndex;
        int n = B.numRows;
        X = new CMatrix(B.shape);
        det = U.data[U.data.length-1];

        xCol = new Complex128[n];

        for(int j=0; j<B.numCols; j++) {
            X.data[(n-1)*X.numCols + j] = B.data[(n-1)*X.numCols + j].div(U.data[n*n-1]);
            det = det.mult(U.data[j*(n+1)]);

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++)
                xCol[k] = X.data[k*X.numCols + j];

            for(int i=n-2; i>-1; i--) {
                sum = Complex128.ZERO;
                uIndex = i*U.numCols;
                xIndex = i*X.numCols + j;
                diag = U.data[i*(n+1)];

                if(j==0) det = det.mult(diag);

                for(int k=i+1; k<n; k++)
                    sum = sum.add(U.data[uIndex + k].mult(xCol[k]));

                Complex128 value = B.data[xIndex].sub(sum).div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves the linear system of equations given by U*X=I where the coefficient matrix U
     * is an {@link CMatrix#isTriU() upper triangular} matrix and I is the {@link CMatrix#isI() identity}
     * matrix of appropriate size.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not actually
     *          upper triangular, it will be treated as if it were.
     * @return The solution to X in the linear system U*X=B.
     * @throws SingularMatrixException If the matrix {@code U} is singular (i.e. has a zero on the principle diagonal).
     */
    public CMatrix solveIdentity(CMatrix U) {
        checkParams(U, U.shape);

        Complex128 sum, diag;
        int uIndex, xIndex;
        int n = U.numRows;
        X = new CMatrix(U.shape);
        det = U.data[U.data.length-1];

        xCol = new Complex128[n];
        X.data[X.data.length-1] = det.multInv();

        for(int j=0; j<U.numCols; j++) {

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=n-2; i>-1; i--) {
                sum = (i == j) ? Complex128.ONE : Complex128.ZERO;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                uIndex += i+1;
                diag = U.data[i*(n+1)];

                if(j==0) det = det.mult(diag);

                for(int k=i+1; k<n; k++) {
                    sum = sum.sub(U.data[uIndex++].mult(xCol[k]));
                }

                Complex128 value = sum.div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }


    /**
     * Solves a special case of the linear system U*X=L for X where the coefficient matrix U
     * is an {@link CMatrix#isTriU() upper triangular} matrix and the constant matrix L is
     * {@link CMatrix#isTriL() lower triangular}.
     *
     * @param U Upper triangular coefficient matrix
     * @param L Lower triangular constant matrix.
     * @return The result of solving the linear system U*X=L for the matrix X.
     */
    public CMatrix solveLower(CMatrix U, CMatrix L) {
        checkParams(U, L.shape);

        Complex128 sum, diag;
        int uIndex, xIndex;
        int n = L.numRows;
        Complex128 uValue = U.data[n*n-1];
        int rowOffset = (n-1)*n;
        X = new CMatrix(L.shape);
        det = U.data[U.data.length-1];

        xCol = new Complex128[n];

        for(int j=0; j<n; j++) {
            X.data[rowOffset] = L.data[rowOffset++].div(uValue);

            // Store column to improve cache performance on innermost loop.
            for(int k=0; k<n; k++) {
                xCol[k] = X.data[k*X.numCols + j];
            }

            for(int i=L.numCols-2; i>=0; i--) {
                sum = Complex128.ZERO;
                uIndex = i*U.numCols;
                xIndex = uIndex + j;
                diag = U.data[i*(n+1)];

                if(j==0) det = det.mult(diag);

                for(int k=i+1; k<n; k++) {
                    sum = sum.add(U.data[uIndex + k].mult(xCol[k]));
                }

                Complex128 value = L.data[xIndex].sub(sum).div(diag);
                X.data[xIndex] = value;
                xCol[i] = value;
            }
        }

        checkSingular(det.mag(), U.numRows, U.numCols); // Ensure the matrix is not singular.

        return X;
    }
}
