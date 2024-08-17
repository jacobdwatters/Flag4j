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

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.linalg.decompositions.schur.ComplexSchur;
import org.flag4j.linalg.decompositions.schur.RealSchur;
import org.flag4j.linalg.decompositions.schur.Schur;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.operations_old.common.real.AggregateReal;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;

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
    public static CVectorOld get2x2EigenValues(MatrixOld src) {
        ParameterChecks.assertEquals(2, src.numRows, src.numCols);
        return new CVectorOld(get2x2EigenValues(src.entries[0], src.entries[1], src.entries[2], src.entries[3]));
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
            Arrays.fill(lambda, CNumber.ZERO);
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
    public static CVectorOld get2x2EigenValues(CMatrixOld src) {
        ParameterChecks.assertEquals(2, src.numRows, src.numCols);
        return new CVectorOld(get2x2EigenValues(src.entries[0], src.entries[1], src.entries[2], src.entries[3]));
    }


    public static CNumber[] get2x2EigenValues(CNumber a11, CNumber a12, CNumber a21, CNumber a22) {
        // Initialize eigenvalues array
        CNumber[] lambda = new CNumber[2];

        // Compute maximum magnitude for scaling
        double maxAbs = Math.max(Math.max(a11.mag(), a12.mag()), Math.max(a21.mag(), a22.mag()));

        if(maxAbs == 0) {
            lambda[0] = CNumber.ZERO;
            lambda[1] = CNumber.ZERO;
            return lambda;
        }

        // Scale the matrix to avoid overflow/underflow
        a11 = a11.div(maxAbs);
        a12 = a12.div(maxAbs);
        a21 = a21.div(maxAbs);
        a22 = a22.div(maxAbs);

        // Trace and determinant for the 2x2 matrix
        CNumber trace = a11.add(a22);
        CNumber det = a11.mult(a22).sub(a12.mult(a21));

        // Compute the middle term of the quadratic equation
        CNumber middleTerm = trace.mult(trace).div(4).sub(det);

        // Compute the square root of the middle term
        CNumber sqrtMiddle = CNumber.sqrt(middleTerm);

        // Compute eigenvalues
        lambda[0] = trace.div(2).add(sqrtMiddle);
        lambda[1] = trace.div(2).sub(sqrtMiddle);

        // Scale back the eigenvalues
        lambda[0] = lambda[0].mult(new CNumber(maxAbs));
        lambda[1] = lambda[1].mult(new CNumber(maxAbs));

        return lambda;
    }


    /**
     * Computes the eigenvalues of a square real dense matrix. For reproducibility see {@link #getEigenValues(MatrixOld, long)}. If the
     * algorithm fails to converge within the default maximum number of iterations, {@link #getEigenValues(MatrixOld, long, int)} can
     * be used to specify a larger number of iterations to attempt.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(MatrixOld, long)
     * @see #getEigenValues(MatrixOld, long, int)
     */
    public static CVectorOld getEigenValues(MatrixOld src) {
        RealSchur schur = new RealSchur(false);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a square real dense matrix. If the algorithm fails to converge within the default maximum number
     * of iterations, {@link #getEigenValues(MatrixOld, long, int)} can be used to specify a larger number of iterations to attempt.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(MatrixOld)
     * @see #getEigenValues(MatrixOld, long, int)
     */
    public static CVectorOld getEigenValues(MatrixOld src, long seed) {
        RealSchur schur = new RealSchur(false, seed);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a square real dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(MatrixOld)
     * @see #getEigenValues(MatrixOld, long)
     */
    public static CVectorOld getEigenValues(MatrixOld src, long seed, int maxIterationFactor) {
        RealSchur schur = new RealSchur(false, seed).setMaxIterationFactor(maxIterationFactor);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a real dense square matrix.
     * @param src MatrixOld to compute eigenvalues of.
     * @param schur Schur decomposer to use in the eigenvalue computation.
     * @return The eigenvalues of the {@code src} matrix stored in a complex vector ({@link CVectorOld}).
     */
    private static CVectorOld getEigenValues(MatrixOld src, RealSchur schur) {
        CVectorOld lambdas = new CVectorOld(src.numRows);
        MatrixOld T = schur.decompose(src).getT();

        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = new CNumber(T.entries[m*numRows + m]);
            } else {
                double a11 = T.entries[m*numRows + m];
                double a12 = T.entries[m*numRows + m + 1];
                double a21 = T.entries[(m+1)*numRows + m];
                double a22 = T.entries[(m+1)*numRows + m + 1];

                if(Math.abs(a21) > EPS_F64*(Math.abs(a11) + Math.abs(a22))) {
                    // Non-converged 2x2 block found.
                    CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                    lambdas.entries[m] = mu[0];
                    lambdas.entries[++m] = mu[1];
                } else {
                    lambdas.entries[m] = new CNumber(a11);
                }
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix. For reproducibility see {@link #getEigenValues(CMatrixOld, long)}.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(CMatrixOld, long)
     */
    public static CVectorOld getEigenValues(CMatrixOld src) {
        return getEigenValues(src, new ComplexSchur(false));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues. This allows for reproducibility despite randomness in the
     * eigenvalue computation algorithm.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(CMatrixOld)
     */
    public static CVectorOld getEigenValues(CMatrixOld src, long seed) {
        return getEigenValues(src, new ComplexSchur(false, seed));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(MatrixOld)
     * @see #getEigenValues(MatrixOld, long)
     */
    public static CVectorOld getEigenValues(CMatrixOld src, long seed, int maxIterationFactor) {
        return getEigenValues(src,
                new ComplexSchur(false, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param schur The schur decomposer to use in the eigenvalue decomposition.
     * @return The eigenvalues of the {@code src} matrix.
     */
    private static CVectorOld getEigenValues(CMatrixOld src, ComplexSchur schur) {
        CVectorOld lambdas = new CVectorOld(src.numRows);
        CMatrixOld T = schur.decompose(src).getT();
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
                    lambdas.entries[m] = a11;
                }
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(MatrixOld, long)
     * @see #getEigenVectors(MatrixOld, long, int)
     * @see #getEigenValues(MatrixOld)
     * @see #getEigenValues(MatrixOld, long)
     * @see #getEigenValues(MatrixOld, long, int)
     */
    public static CMatrixOld getEigenVectors(MatrixOld src) {
        return getEigenVectors(src, new RealSchur(true));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(MatrixOld)
     * @see #getEigenVectors(MatrixOld, long, int)
     * @see #getEigenValues(MatrixOld)
     * @see #getEigenValues(MatrixOld, long)
     * @see #getEigenValues(MatrixOld, long, int)
     */
    public static CMatrixOld getEigenVectors(MatrixOld src, long seed) {
        return getEigenVectors(src, new RealSchur(true, seed));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrixOld)
     * @see #getEigenVectors(CMatrixOld, long)
     * @see #getEigenValues(CMatrixOld)
     * @see #getEigenValues(CMatrixOld, long)
     * @see #getEigenValues(CMatrixOld, long, int)
     */
    public static CMatrixOld getEigenVectors(MatrixOld src, long seed, int maxIterationFactor) {
        return getEigenVectors(src,
                new RealSchur(true, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @param schur Schur decomposer to use in the computation of the eigenvectors.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    private static CMatrixOld getEigenVectors(MatrixOld src, RealSchur schur) {
        schur.decompose(src);
        CMatrixOld[] complexTU = schur.real2ComplexSchur();
        CMatrixOld U = complexTU[1];

        if(src.isSymmetric()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrixOld Q = getEigenVectorsTriu(complexTU[0]); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrixOld, long)
     * @see #getEigenVectors(CMatrixOld, long, int)
     * @see #getEigenValues(CMatrixOld)
     * @see #getEigenValues(CMatrixOld, long)
     * @see #getEigenValues(CMatrixOld, long, int)
     */
    public static CMatrixOld getEigenVectors(CMatrixOld src) {
        return getEigenVectors(src, new ComplexSchur(true));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrixOld)
     * @see #getEigenVectors(CMatrixOld, long, int)
     * @see #getEigenValues(CMatrixOld)
     * @see #getEigenValues(CMatrixOld, long)
     * @see #getEigenValues(CMatrixOld, long, int)
     */
    public static CMatrixOld getEigenVectors(CMatrixOld src, long seed) {
        return getEigenVectors(src, new ComplexSchur(true, seed));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for during eigenvalue computation. The maximum number of iterations will be computed as:
     * <pre>
     *      {@code maxIteration = maxIterationFactor * src.numRows;}</pre>
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrixOld)
     * @see #getEigenVectors(CMatrixOld, long)
     * @see #getEigenValues(CMatrixOld)
     * @see #getEigenValues(CMatrixOld, long)
     * @see #getEigenValues(CMatrixOld, long, int)
     */
    public static CMatrixOld getEigenVectors(CMatrixOld src, long seed, int maxIterationFactor) {
        return getEigenVectors(src,
                new ComplexSchur(true, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvectors of a square complex dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @param schur Schur decomposer to use when computing the eigenvectors.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    private static CMatrixOld getEigenVectors(CMatrixOld src, ComplexSchur schur) {
        schur.decompose(src);
        CMatrixOld T;
        CMatrixOld U;

        if(src.numRows == 2) {
            CMatrixOld[] complexTU = schur.real2ComplexSchur();
            T = complexTU[0];
            U = complexTU[1];
        } else {
            T = schur.getT();
            U = schur.getU();
        }

        if(src.isHermitian()) {
            // Then the columns of U are the complete orthonormal set of eigenvectors of the src matrix.
            return U;
        } else {
            // For a non-symmetric matrix, only the first column of U will be an eigenvector of the src matrix.
            CMatrixOld Q = getEigenVectorsTriu(T); // Compute the eigenvectors of T.
            return U.mult(Q); // Convert the eigenvectors of T to the eigenvectors of the src matrix.
        }
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static MatrixOld getEigenVectorsTriu(MatrixOld T) {
        RealBackSolver backSolver = new RealBackSolver();
        MatrixOld Q = new MatrixOld(T.numRows);

        MatrixOld S_hat;
        VectorOld r;
        VectorOld v;

        for(int j=0; j<T.numRows; j++) {
            S_hat = new MatrixOld(j, j);
            r = new VectorOld(j);
            makeSystem(T, j, S_hat, r);

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r);
                v = v.join(new VectorOld(1.0));
            } else {
                v = new VectorOld(1.0);
            }

            v = v.normalize().join(new VectorOld(T.numRows-v.size));
            Q.setCol(v, j);
        }

        return Q;
    }


    /**
     * Computes the eigenvectors of an upper triangular matrix.
     * @param T The upper triangular matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code T} as its columns.
     */
    public static CMatrixOld getEigenVectorsTriu(CMatrixOld T) {
        ComplexBackSolver backSolver = new ComplexBackSolver();
        CMatrixOld Q = new CMatrixOld(T.numRows);

        CMatrixOld S_hat;
        CVectorOld r;
        CVectorOld v;

        for(int j=0; j<T.numRows; j++) {
            S_hat = CMatrixOld.getEmpty(j, j);
            r = CVectorOld.getEmpty(j);
            makeSystem(T, j, S_hat, r);

            if(S_hat.entries.length > 0) {
                v = backSolver.solve(S_hat, r);
                v = v.join(new CVectorOld(1.0));
            } else {
                v = new CVectorOld(1.0);
            }

            v = v.normalize().join(new VectorOld(T.numRows-v.size));
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
    private static void makeSystem(CMatrixOld T, int j, CMatrixOld S_hat, CVectorOld r) {
        CNumber lam = T.entries[j*T.numCols + j];

        // Copy values from T and subtract eigenvalue from diagonal.
        for(int i=0; i<j; i++) {
            int tRow = i*T.numRows;
            int diffRow = i*j;
            System.arraycopy(T.entries, tRow, S_hat.entries, diffRow, j);
            S_hat.entries[diffRow + i] = S_hat.entries[diffRow + i].sub(lam);
            r.entries[i] = T.entries[tRow + j].mult(-1);
        }
    }


    /**
     * Constructs the matrix {@code (T - lambda*I)[0:j][0:j]} for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue {@code lambda}.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue {@code lambda} appears.
     */
    private static void makeSystem(MatrixOld T, int j, MatrixOld S_hat, VectorOld r) {
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
    public static CMatrixOld[] getEigenPairs(MatrixOld src) {
        int numRows = src.numRows;
        CMatrixOld lambdas = new CMatrixOld(1, src.numRows);

        RealSchur schur = new RealSchur(true).decompose(src);
        CMatrixOld[] complexTU = schur.real2ComplexSchur();
        CMatrixOld T = complexTU[0];
        CMatrixOld U = complexTU[1];

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++) {
            lambdas.entries[i] = T.entries[i*numRows + i];
        }

        // If the source matrix is symmetric, then U will contain its eigenvectors.
        if(!src.isSymmetric()) {
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
        }

        return new CMatrixOld[]{lambdas, U};
    }


    /**
     * Computes the eigenvalues and eigenvectors of a square real matrix.
     * @param src The matrix to compute the eigenvalues and vectors of.
     * @return An array containing two matrices. The first matrix has shape {@code 1xsrc.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRowsxsrc.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrixOld[] getEigenPairs(CMatrixOld src) {
        CMatrixOld lambdas = new CMatrixOld(1, src.numRows);

        Schur<CMatrixOld, CNumber[]> schur = new ComplexSchur(true).decompose(src);
        CMatrixOld T = schur.getT();
        CMatrixOld U = schur.getU();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++) {
            lambdas.entries[i] = T.entries[i*numRows + i];
        }

        // If the src matrix is hermitian, then U will contain the eigenvectors.
        if(!src.isHermitian()) {
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
        }

        return new CMatrixOld[]{lambdas, U};
    }
}
