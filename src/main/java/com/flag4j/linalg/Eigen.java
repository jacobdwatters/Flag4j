/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package com.flag4j.linalg;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CMatrix;
import com.flag4j.dense.CVector;
import com.flag4j.dense.Matrix;
import com.flag4j.dense.Vector;
import com.flag4j.linalg.decompositions.schur.ComplexSchurDecomposition;
import com.flag4j.linalg.decompositions.schur.RealSchurDecomposition;
import com.flag4j.linalg.decompositions.schur.SchurDecomposition;
import com.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import com.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.util.ParameterChecks;

import static com.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * This class provides several methods useful for computing eigen values, eigen vectors, as well as singular values and
 * singular vectors.
 */
public class Eigen {

    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(Matrix src) {
        ParameterChecks.assertEquals(2, src.numRows, src.numCols);
        return new CVector(get2x2EigenValues(src.entries[0], src.entries[1], src.entries[2], src.entries[3]));
    }


    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param a11 First entry in matrix (at index (0, 0)).
     * @param a12 Second entry in matrix (at index (0, 1)).
     * @param a21 Third entry in matrix (at index (1, 0)).
     * @param a22 Fourth entry in matrix (at index (1, 1)).
     * @return A complex array containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CNumber[] get2x2EigenValues(double a11, double a12, double a21, double a22) {
        // This method computes eigenvalues in a stable way which is more resilient to overflow errors than
        // standard methods.
        CNumber[] lambda = new CNumber[2];
        double maxAbs = AggregateReal.maxAbs(a11, a12, a21, a22);

        if(maxAbs == 0) {
            return lambda;
        } else {
            a11 /= maxAbs;
            a12 /= maxAbs;
            a21 /= maxAbs;
            a22 /= maxAbs;

            double c;
            double s;

            if (a12 + a21 == 0) {
                c = s = 1.0 / Math.sqrt(2);
            } else {
                double aa = (a11 - a22);
                double bb = (a12 + a21);

                double t_hat = aa/bb;
                double t = t_hat/(1.0 + Math.sqrt(1.0 + t_hat*t_hat));

                c = 1.0 / Math.sqrt(1.0 + t*t);
                s = c*t;
            }

            double c2 = c*c;
            double s2 = s*s;
            double cs = c*s;

            double b11 = c2*a11 + s2*a22 - cs*(a12 + a21);
            double b12 = c2*a12 - s2*a21 + cs*(a11 - a22);
            double b21 = c2*a21 - s2*a12 + cs*(a11 - a22);

            // apply second rotator to make A upper triangular if real eigenvalues
            if (b21*b12 >= 0) {
                if (b12 == 0) {
                    c = 0;
                    s = 1;
                } else {
                    s = Math.sqrt(b21/(b12 + b21));
                    c = Math.sqrt(b12/(b12 + b21));
                }

                cs = c*s;

                a11 = b11 - cs*(b12 + b21);
                a22 = b11 + cs*(b12 + b21);

                lambda[0] = new CNumber(a11*maxAbs);
                lambda[1] = new CNumber(a22*maxAbs);
            } else {
                double im = Math.sqrt(-b21*b12);

                lambda[0] = new CNumber(b11*maxAbs, im*maxAbs);
                lambda[1] = new CNumber(b11*maxAbs, -im*maxAbs);
            }
        }

        return lambda;
    }


    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(CMatrix src) {
        ParameterChecks.assertEquals(2, src.numRows, src.numCols);
        return new CVector(get2x2EigenValues(src.entries[0], src.entries[1], src.entries[2], src.entries[3]));
    }


    public static CNumber[] get2x2EigenValues(CNumber a11, CNumber a12, CNumber a21, CNumber a22) {
        // Initialize eigenvalues array
        CNumber[] lambda = new CNumber[2];

        // Compute maximum magnitude for scaling
        double maxAbs = Math.max(Math.max(a11.mag(), a12.mag()), Math.max(a21.mag(), a22.mag()));

        if(maxAbs == 0) {
            lambda[0] = new CNumber(0, 0);
            lambda[1] = new CNumber(0, 0);
            return lambda;
        }

        // Scale the matrix to avoid overflow/underflow
        a11 = a11.div(new CNumber(maxAbs, 0));
        a12 = a12.div(new CNumber(maxAbs, 0));
        a21 = a21.div(new CNumber(maxAbs, 0));
        a22 = a22.div(new CNumber(maxAbs, 0));

        // Trace and determinant for the 2x2 matrix
        CNumber trace = a11.add(a22);
        CNumber det = a11.mult(a22).sub(a12.mult(a21));

        // Compute the middle term of the quadratic equation
        CNumber middleTerm = trace.mult(trace).div(new CNumber(4, 0)).sub(det);

        // Compute the square root of the middle term
        CNumber sqrtMiddle = CNumber.sqrt(middleTerm);

        // Compute eigenvalues
        lambda[0] = trace.div(new CNumber(2)).add(sqrtMiddle);
        lambda[1] = trace.div(new CNumber(2)).sub(sqrtMiddle);

        // Scale back the eigenvalues
        lambda[0] = lambda[0].mult(new CNumber(maxAbs, 0));
        lambda[1] = lambda[1].mult(new CNumber(maxAbs, 0));

        return lambda;
    }


    /**
     * Computes the eigenvalues for the lower right 2x2 block matrix of a larger matrix.
     * @param src Source matrix to compute eigenvalues of lower right 2x2 block.
     * @return A vector of length 2 containing the eigenvalues of the lower right 2x2 block of {@code src}.
     */
    public static CVector get2x2LowerRightBlockEigenValues(CMatrix src) {
        int n = src.numRows-1;

        return get2x2EigenValues(
                new CMatrix(new CNumber[][]{
                    {src.entries[(n-1)*(src.numCols + 1)], src.entries[(n-1)*src.numCols + n]},
                    {src.entries[n*(src.numCols + 1) - 1], src.entries[n*(src.numCols + 1)]}
                })
        );
    }


    /**
     * Computes the eigenvalues for the lower right 2x2 block matrix of a larger matrix.
     * @param src Source matrix to compute eigenvalues of lower right 2x2 block.
     * @return A vector of length 2 containing the eigenvalues of the lower right 2x2 block of {@code src}.
     */
    public static CVector get2x2LowerRightBlockEigenValues(Matrix src) {
        int n = src.numRows-1;

        return get2x2EigenValues(
                new Matrix(new double[][]{
                        {src.entries[(n-1)*(src.numCols + 1)], src.entries[(n-1)*src.numCols + n]},
                        {src.entries[n*(src.numCols + 1) - 1], src.entries[n*(src.numCols + 1)]}
                })
        );
    }


    /**
     * Computes the eigenvalues of a square real dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     */
    public static CVector getEigenValues(Matrix src) {
        CVector lambdas = new CVector(src.numRows);

        SchurDecomposition<Matrix, double[]> schur = new RealSchurDecomposition(false).decompose(src);
        Matrix T = schur.getT();

        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            double a11 = T.entries[m*numRows + m];
            double a12 = T.entries[m*numRows + m + 1];
            double a21 = T.entries[(m+1)*numRows + m];
            double a22 = T.entries[(m+1)*numRows + m  +1];

            if(Math.abs(a21) > EPS_F64*(Math.abs(a11) + Math.abs(a22))) {
                // Non-converged 2x2 block found.
                CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                lambdas.entries[m] = mu[0];
                lambdas.entries[++m] = mu[1];
            } else {
                lambdas.entries[m] = new CNumber(a22);
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     */
    public static CVector getEigenValues(CMatrix src) {
        CVector lambdas = new CVector(src.numRows);

        SchurDecomposition<CMatrix, CNumber[]> schur = new ComplexSchurDecomposition(false).decompose(src);
        CMatrix T = schur.getT();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = T.entries[m*numRows + m];
            } else {
                CNumber a11 = T.entries[m*numRows + m];
                CNumber a12 = T.entries[m*numRows + m + 1];
                CNumber a21 = T.entries[(m+1)*numRows + m];
                CNumber a22 = T.entries[(m+1)*numRows + m  +1];

                if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                    // Non-converged 2x2 block found.
                    CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                    lambdas.entries[m] = mu[0];
                    lambdas.entries[++m] = mu[1];
                } else {
                    lambdas.entries[m] = a22.copy();
                }
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    public static CMatrix getEigenVectors(Matrix src) {
        RealSchurDecomposition schur = new RealSchurDecomposition(true).decompose(src);
        CMatrix[] complexTU = schur.real2ComplexSchur();
        CMatrix U = complexTU[1];

        if(src.isSymmetric()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrix Q = getEigenVectorsTriu(complexTU[0]); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of a square complex dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    public static CMatrix getEigenVectors(CMatrix src) {
        SchurDecomposition<CMatrix, CNumber[]> schur = new ComplexSchurDecomposition(true).decompose(src);
        CMatrix U = schur.getU();

        if(src.isHermitian()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrix Q = getEigenVectorsTriu(schur.getT()); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static CMatrix getEigenVectorsTriu(Matrix T) {
        RealBackSolver backSolver = new RealBackSolver();
        Matrix I = Matrix.I(T.numRows);
        CMatrix Q = new CMatrix(T.numRows);

        for(int j=0; j<T.numRows; j++) {
            Matrix t = I.mult(T.get(j, j)); // TODO: add diag(...) method so this scalar multiplication doesnt need to be computed.
            Matrix S = T.sub(t).getSlice(0, j+1, 0, j+1);
            Matrix S_hat = S.getSlice(0, j, 0, j);
            Vector r = S.getSlice(0, j, S.numCols-1, S.numCols).toVector();
            Vector v;

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r.mult(-1));
                v = v.join(new Vector(1.0));
            } else {
                v = new Vector(1.0);
            }

            v = v.normalize().join(new Vector(T.numRows-v.size));
            Q.setCol(v.entries, j);
        }

        return Q;
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static CMatrix getEigenVectorsTriu(CMatrix T) {
        ComplexBackSolver backSolver = new ComplexBackSolver();
        Matrix I = Matrix.I(T.numRows);
        CMatrix Q = new CMatrix(T.numRows);

        for(int j=0; j<T.numRows; j++) {
            CMatrix t = I.mult(T.get(j, j)); // TODO: add diag(...) method so this scalar multiplication doesnt need to be computed.
            CMatrix S = T.sub(t).getSlice(0, j+1, 0, j+1);
            CMatrix S_hat = S.getSlice(0, j, 0, j);
            CVector r = S.getSlice(0, j, S.numCols-1, S.numCols).toVector();
            CVector v;

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r.mult(-1));
                v = v.join(new CVector(1.0));
            } else {
                v = new CVector(1.0);
            }

            v = v.normalize().join(new Vector(T.numRows-v.size));
            Q.setCol(v, j);
        }

        return Q;
    }


    /**
     * Computes the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and vectors of.
     * @return An array containing two matrices. The first matrix has shape {@code 1xsrc.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRowsxsrc.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrix[] getEigenPairs(Matrix src) {
        CMatrix lambdas = new CMatrix(1, src.numRows);

        RealSchurDecomposition schur = new RealSchurDecomposition(true).decompose(src);
        CMatrix[] complexTU = schur.real2ComplexSchur();
        Matrix tReal = schur.getT();
        CMatrix T = complexTU[0];
        CMatrix U = complexTU[1];
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            double a11 = tReal.entries[m*numRows + m];
            double a12 = tReal.entries[m*numRows + m + 1];
            double a21 = tReal.entries[(m+1)*numRows + m];
            double a22 = tReal.entries[(m+1)*numRows + m  +1];

            if(Math.abs(a21) > EPS_F64*(Math.abs(a11) + Math.abs(a22))) {
                // Non-converged 2x2 block found.
                CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                lambdas.entries[m] = mu[0];
                lambdas.entries[++m] = mu[1];
            } else {
                lambdas.entries[m] = new CNumber(a22);
            }
        }

        if(!src.isSymmetric()) {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
        }

        return new CMatrix[]{lambdas, U};
    }


    /**
     * Computes the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and vectors of.
     * @return An array containing two matrices. The first matrix has shape {@code 1xsrc.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRowsxsrc.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrix[] getEigenPairs(CMatrix src) {
        CMatrix lambdas = new CMatrix(1, src.numRows);

        SchurDecomposition<CMatrix, CNumber[]> schur = new ComplexSchurDecomposition(true).decompose(src);
        CMatrix T = schur.getT();
        CMatrix U = schur.getU();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = T.entries[m*numRows + m];
            } else {
                CNumber a11 = T.entries[m*numRows + m];
                CNumber a12 = T.entries[m*numRows + m + 1];
                CNumber a21 = T.entries[(m+1)*numRows + m];
                CNumber a22 = T.entries[(m+1)*numRows + m  +1];

                if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                    // Non-converged 2x2 block found.
                    CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                    lambdas.entries[m] = mu[0];
                    lambdas.entries[++m] = mu[1];
                } else {
                    lambdas.entries[m] = a22.copy();
                }
            }
        }

        if(!src.isHermitian()) {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
        }

        return new CMatrix[]{lambdas, U};
    }
}
