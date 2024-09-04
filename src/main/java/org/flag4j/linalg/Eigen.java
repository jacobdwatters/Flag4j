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

package org.flag4j.linalg;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.schur.ComplexSchur;
import org.flag4j.linalg.decompositions.schur.RealSchur;
import org.flag4j.linalg.decompositions.schur.Schur;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.operations_old.common.real.AggregateReal;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * This class provides several methods useful for computing eigenvalues and eigenvectors.
 */
public final class Eigen {


    private Eigen() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(Matrix src) {
        ParameterChecks.ensureEquals(2, src.numRows, src.numCols);
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
    public static Complex128[] get2x2EigenValues(double a11, double a12, double a21, double a22) {
        // This method computes eigenvalues in a stable way which is more resilient to overflow errors than
        // standard methods.
        Complex128[] lambda = new Complex128[2];
        double maxAbs = AggregateReal.maxAbs(a11, a12, a21, a22);

        if(maxAbs == 0) {
            Arrays.fill(lambda, Complex128.ZERO);
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

                lambda[0] = new Complex128(a11*maxAbs);
                lambda[1] = new Complex128(a22*maxAbs);
            } else {
                double im = Math.sqrt(-b21*b12);

                lambda[0] = new Complex128(b11*maxAbs, im*maxAbs);
                lambda[1] = new Complex128(b11*maxAbs, -im*maxAbs);
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
        ParameterChecks.ensureEquals(2, src.numRows, src.numCols);
        return new CVector(get2x2EigenValues(src.entries[0], src.entries[1], src.entries[2], src.entries[3]));
    }


    /**
     * Computes the eigenvalues of a 2-by-2 complex matrix (assumed to be row major).
     * @param a11 The first entry in the 2-by-2 complex matrix
     * @param a12 The second entry in the 2-by-2 complex matrix
     * @param a21 The third entry in the 2-by-2 complex matrix
     * @param a22 The fourth entry in the 2-by-2 complex matrix
     * @return An array containing the eigenvalues of the specified 2-by-2 complex matrix. Eigenvalues will be repeated per their
     * multiplicity.
     */
    public static Complex128[] get2x2EigenValues(Complex128 a11, Complex128 a12, Complex128 a21, Complex128 a22) {
        Complex128[] lambda = new Complex128[2];

        // Compute maximum magnitude for scaling.
        double maxAbs = Math.max(Math.max(a11.mag(), a12.mag()), Math.max(a21.mag(), a22.mag()));

        if(maxAbs == 0) {
            lambda[0] = Complex128.ZERO;
            lambda[1] = Complex128.ZERO;
            return lambda;
        }

        // Scale the matrix to avoid over/underflow.
        a11 = a11.div(maxAbs);
        a12 = a12.div(maxAbs);
        a21 = a21.div(maxAbs);
        a22 = a22.div(maxAbs);

        // Trace and determinant for the 2x2 matrix.
        Complex128 trace = a11.add(a22);
        Complex128 det = a11.mult(a22).sub(a12.mult(a21));

        // Compute the middle term of the quadratic equation.
        Complex128 middleTerm = trace.mult(trace).div(4).sub(det);

        // Compute the square root of the middle term.
        Complex128 sqrtMiddle = middleTerm.sqrt();

        // Compute eigenvalues.
        lambda[0] = trace.div(2).add(sqrtMiddle);
        lambda[1] = trace.div(2).sub(sqrtMiddle);

        // Scale back the eigenvalues.
        lambda[0] = lambda[0].mult(new Complex128(maxAbs));
        lambda[1] = lambda[1].mult(new Complex128(maxAbs));

        return lambda;
    }


    /**
     * Computes the eigenvalues of a square real dense matrix. For reproducibility see {@link #getEigenValues(Matrix, long)}. If the
     * algorithm fails to converge within the default maximum number of iterations, {@link #getEigenValues(Matrix, long, int)} can
     * be used to specify a larger number of iterations to attempt.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(Matrix, long)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CVector getEigenValues(Matrix src) {
        RealSchur schur = new RealSchur(false);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a square real dense matrix. If the algorithm fails to converge within the default maximum number
     * of iterations, {@link #getEigenValues(Matrix, long, int)} can be used to specify a larger number of iterations to attempt.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CVector getEigenValues(Matrix src, long seed) {
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
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     */
    public static CVector getEigenValues(Matrix src, long seed, int maxIterationFactor) {
        RealSchur schur = new RealSchur(false, seed).setMaxIterationFactor(maxIterationFactor);
        return getEigenValues(src, schur);
    }


    /**
     * Computes the eigenvalues of a real dense square matrix.
     * @param src Matrix to compute eigenvalues of.
     * @param schur Schur decomposer to use in the eigenvalue computation.
     * @return The eigenvalues of the {@code src} matrix stored in a complex vector ({@link CVector}).
     */
    private static CVector getEigenValues(Matrix src, RealSchur schur) {
        CVector lambdas = new CVector(src.numRows);
        Matrix T = schur.decompose(src).getT();

        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = new Complex128(T.entries[m*numRows + m]);
            } else {
                double a11 = T.entries[m*numRows + m];
                double a12 = T.entries[m*numRows + m + 1];
                double a21 = T.entries[(m+1)*numRows + m];
                double a22 = T.entries[(m+1)*numRows + m + 1];

                if(Math.abs(a21) > EPS_F64*(Math.abs(a11) + Math.abs(a22))) {
                    // Non-converged 2x2 block found.
                    Complex128[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                    lambdas.entries[m] = mu[0];
                    lambdas.entries[++m] = mu[1];
                } else {
                    lambdas.entries[m] = new Complex128(a11);
                }
            }
        }

        return lambdas;
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix. For reproducibility see {@link #getEigenValues(CMatrix, long)}.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(CMatrix, long)
     */
    public static CVector getEigenValues(CMatrix src) {
        return getEigenValues(src, new ComplexSchur(false));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param seed Seed for random shifts used in computing the eigenvalues. This allows for reproducibility despite randomness in the
     * eigenvalue computation algorithm.
     * @return The eigenvalues of the {@code src} matrix.
     * @see #getEigenValues(CMatrix)
     */
    public static CVector getEigenValues(CMatrix src, long seed) {
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
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     */
    public static CVector getEigenValues(CMatrix src, long seed, int maxIterationFactor) {
        return getEigenValues(src,
                new ComplexSchur(false, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvalues of a square complex dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @param schur The schur decomposer to use in the eigenvalue decomposition.
     * @return The eigenvalues of the {@code src} matrix.
     */
    private static CVector getEigenValues(CMatrix src, ComplexSchur schur) {
        CVector lambdas = new CVector(src.numRows);
        CMatrix T = schur.decompose(src).getT();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int m=0; m<numRows; m++) {
            if(m == numRows-1) {
                lambdas.entries[m] = T.entries[m*numRows + m];
            } else {
                Complex128 a11 = T.entries[m*numRows + m];
                Complex128 a12 = T.entries[m*numRows + m + 1];
                Complex128 a21 = T.entries[(m+1)*numRows + m];
                Complex128 a22 = T.entries[(m+1)*numRows + m  +1];

                if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                    // Non-converged 2x2 block found.
                    Complex128[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
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
     * @see #getEigenVectors(Matrix, long)
     * @see #getEigenVectors(Matrix, long, int)
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CMatrix getEigenVectors(Matrix src) {
        return getEigenVectors(src, new RealSchur(true));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(Matrix)
     * @see #getEigenVectors(Matrix, long, int)
     * @see #getEigenValues(Matrix)
     * @see #getEigenValues(Matrix, long)
     * @see #getEigenValues(Matrix, long, int)
     */
    public static CMatrix getEigenVectors(Matrix src, long seed) {
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
     * @see #getEigenVectors(CMatrix)
     * @see #getEigenVectors(CMatrix, long)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(Matrix src, long seed, int maxIterationFactor) {
        return getEigenVectors(src,
                new RealSchur(true, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @param schur Schur decomposer to use in the computation of the eigenvectors.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    private static CMatrix getEigenVectors(Matrix src, RealSchur schur) {
        schur.decompose(src);
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
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrix, long)
     * @see #getEigenVectors(CMatrix, long, int)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(CMatrix src) {
        return getEigenVectors(src, new ComplexSchur(true));
    }


    /**
     * Computes the eigenvectors of a square real dense matrix. This method accepts a seed for the random number generator involved in
     * the algorithm for computing the eigenvalues. This allows for reproducibility between calls.
     * @param src The matrix to compute the eigenvectors of.
     * @param seed Seed for random shifts used in the algorithm to compute the eigenvalues of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     * @see #getEigenVectors(CMatrix)
     * @see #getEigenVectors(CMatrix, long, int)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(CMatrix src, long seed) {
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
     * @see #getEigenVectors(CMatrix)
     * @see #getEigenVectors(CMatrix, long)
     * @see #getEigenValues(CMatrix)
     * @see #getEigenValues(CMatrix, long)
     * @see #getEigenValues(CMatrix, long, int)
     */
    public static CMatrix getEigenVectors(CMatrix src, long seed, int maxIterationFactor) {
        return getEigenVectors(src,
                new ComplexSchur(true, seed).setMaxIterationFactor(maxIterationFactor));
    }


    /**
     * Computes the eigenvectors of a square complex dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @param schur Schur decomposer to use when computing the eigenvectors.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    private static CMatrix getEigenVectors(CMatrix src, ComplexSchur schur) {
        schur.decompose(src);
        CMatrix T;
        CMatrix U;

        if(src.numRows == 2) {
            CMatrix[] complexTU = schur.real2ComplexSchur();
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
            CMatrix Q = getEigenVectorsTriu(T); // Compute the eigenvectors of T.
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
                v = v.join(new CVector(Complex128.ONE));
            } else {
                v = new CVector(Complex128.ONE);
            }

            v = v.normalize().join(new CVector(T.numRows-v.size));
            Q.setCol(v, j);
        }

        return Q;
    }


    /**
     * Constructs the matrix (T - &lambda;*I)[0:j][0:j] for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue &lambda;.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue &lambda; appears.
     */
    private static void makeSystem(CMatrix T, int j, CMatrix S_hat, CVector r) {
        Complex128 lam = T.entries[j*T.numCols + j];

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
     * Constructs the matrix (T - &lambda;*I)[0:j][0:j] for use in computing the eigenvalue of the upper triangular matrix
     * {@code T} associated with the j<sup>th</sup> eigenvalue &lambda;.
     * @param T The upper triangular matrix (Assumed to be square).
     * @param j The diagonal index of {@code T} where the eigenvalue &lambda; appears.
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
     * @return An array containing two matrices. The first matrix has shape 1-by-{@code src.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRows}-by-{@code src.numCols} and contains the
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
            lambdas.entries[i] = T.entries[i*numRows + i];
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
     * @return An array containing two matrices. The first matrix has shape 1-by-{@code src.numCols} and contains the eigenvalues
     * of the {@code src} matrix. The second matrix has shape {@code src.numRows}-by-{@code src.numCols} and contains the
     * eigenvectors of the {@code src} matrix as its columns.
     */
    public static CMatrix[] getEigenPairs(CMatrix src) {
        CMatrix lambdas = new CMatrix(1, src.numRows);

        Schur<CMatrix, Complex128[]> schur = new ComplexSchur(true).decompose(src);
        CMatrix T = schur.getT();
        CMatrix U = schur.getU();
        int numRows = src.numRows;

        // Extract eigenvalues of T.
        for(int i=0; i<numRows; i++) {
            lambdas.entries[i] = T.entries[i*numRows + i];
        }

        // If the src matrix is hermitian, then U will contain the eigenvectors.
        if(!src.isHermitian()) {
            U = U.mult(getEigenVectorsTriu(T)); // Compute the eigenvectors of T and convert to eigenvectors of src.
        }

        return new CMatrix[]{lambdas, U};
    }
}
