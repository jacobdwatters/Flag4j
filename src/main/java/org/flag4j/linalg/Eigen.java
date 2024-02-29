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

package org.flag4j.linalg;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Matrix;
import org.flag4j.dense.Vector;
import org.flag4j.linalg.decompositions.schur.ComplexSchur;
import org.flag4j.linalg.decompositions.schur.RealSchur;
import org.flag4j.linalg.decompositions.schur.Schur;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.operations.common.real.AggregateReal;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * This class provides several methods useful for computing eigenvalues and eigenvectors.
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
     * Computes the eigenvalues of a square real dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     */
    public static CVector getEigenValues(Matrix src) {
        CVector lambdas = new CVector(src.numRows);

        Schur<Matrix, double[]> schur = new RealSchur(false).decompose(src);
        Matrix T = schur.getT();

        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = new CNumber(T.entries[m*numRows + m]);
            } else {
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

        Schur<CMatrix, CNumber[]> schur = new ComplexSchur(false).decompose(src);
        CMatrix T = schur.getT();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = T.entries[m*numRows + m].copy();
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
        RealSchur schur = new RealSchur(true).decompose(src);
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
        Schur<CMatrix, CNumber[]> schur = new ComplexSchur(true).decompose(src);
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
    public static Matrix getEigenVectorsTriu(Matrix T) {
        RealBackSolver backSolver = new RealBackSolver();
        Matrix Q = new Matrix(T.numRows);

        Matrix S_hat;
        Vector r;
        Vector v;

        for(int j=0; j<T.numRows; j++) {
            S_hat = new Matrix(j, j);
            r = new Vector(j);
            makeSystem(T, j, S_hat, r);

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r);
                v = v.join(new Vector(1.0));
            } else {
                v = new Vector(1.0);
            }

            v = v.normalize().join(new Vector(T.numRows-v.size));
            Q.setCol(v, j);
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
        CMatrix Q = new CMatrix(T.numRows);

        CMatrix S_hat;
        CVector r;
        CVector v;

        for(int j=0; j<T.numRows; j++) {
            S_hat = CMatrix.getEmpty(j, j);
            r = CVector.getEmpty(j);
            makeSystem(T, j, S_hat, r);

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r);
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
     * Constructs the matrix {@code (T - lambda*I)[0:j][0:j]} for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue {@code lambda}.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue {@code lambda} appears.
     */
    private static void makeSystem(CMatrix T, int j, CMatrix S_hat, CVector r) {
        CNumber lam = T.entries[j*T.numCols + j];

        // Copy values from T and subtract eigenvalue from diagonal.
        for(int i=0; i<j; i++) {
            int tRow = i*T.numRows;
            int diffRow = i*j;
            ArrayUtils.arraycopy(T.entries, tRow, S_hat.entries, diffRow, j);
            S_hat.entries[diffRow + i].subEq(lam);
            r.entries[i] = T.entries[tRow + j].mult(-1);
        }
    }


    /**
     * Constructs the matrix {@code (T - lambda*I)[0:j][0:j]} for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue {@code lambda}.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue {@code lambda} appears.
     */
    private static void makeSystem(Matrix T, int j, Matrix S_hat, Vector r) {
        double lam = T.entries[j*T.numCols + j];

        // Copy values from T and subtract eigenvalue from diagonal.
        for(int i=0; i<j; i++) {
            int tRow = i*T.numRows;
            int diffRow = i*j;
            System.arraycopy(T.entries, tRow, S_hat.entries, diffRow, j);
            S_hat.entries[diffRow + i] -= (lam);
            r.entries[i] = -T.entries[tRow + j];
        }
    }


    /**
     * Computes the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and vectors of.
     * @return An array containing two matrices. The first matrix has shape {@code 1xsrc.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRowsxsrc.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrix[] getEigenPairs(Matrix src) {
        int numRows = src.numRows;
        CMatrix lambdas = new CMatrix(1, src.numRows);

        RealSchur schur = new RealSchur(true).decompose(src);
        CMatrix[] complexTU = schur.real2ComplexSchur();
        CMatrix T = complexTU[0];
        CMatrix U = complexTU[1];

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++) {
            lambdas.entries[i] = T.entries[i*numRows + i].copy();
        }

        // If the source matrix is symmetric, then U will contain its eigenvectors.
        if(!src.isSymmetric()) {
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

        Schur<CMatrix, CNumber[]> schur = new ComplexSchur(true).decompose(src);
        CMatrix T = schur.getT();
        CMatrix U = schur.getU();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++) {
            lambdas.entries[i] = T.entries[i*numRows + i].copy();
        }

        // If the src matrix is hermitian, then U will contain the eigenvectors.
        if(!src.isHermitian()) {
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
        }

        return new CMatrix[]{lambdas, U};
    }

    public static void main(String[] args) {
        Matrix A = new Matrix(new double[][]{
                {0, 0, 0, 1},
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}});

        CMatrix[] eig = Eigen.getEigenPairs(A);
        CVector eigvals = eig[0].toVector();
        CMatrix eigvectors = eig[1];

        System.out.println("Eigenvalues: " + eigvals + "\n");
        System.out.println("Eigenvectors:\n" + eigvectors + "\n");

        for(int j=0; j<eigvectors.numCols; j++) {
            CVector v = eigvectors.getCol(j);
            System.out.println(A.mult(v).sub(v.mult(eigvals.get(j))) + "\n");
        }
    }
}
