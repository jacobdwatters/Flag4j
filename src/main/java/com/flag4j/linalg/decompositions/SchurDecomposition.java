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

package com.flag4j.linalg.decompositions;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.linalg.Eigen;
import com.flag4j.linalg.transformations.Givens;
import com.flag4j.util.ParameterChecks;

import java.util.Random;

/**
 * <p>This abstract class specifies methods for computing the Schur decomposition of a square matrix.
 * That is, decompose a square matrix {@code A} into {@code A=UTU<sup>H</sup>} where {@code U} is a unitary
 * matrix whose columns are the eigenvectors of {@code A} and {@code T} is an upper triangular matrix in
 * Schur form whose diagonal entries are the eigenvalues of {@code A}, corresponding to the columns of {@code U},
 * repeated per their multiplicity.</p>
 *
 * <p>Note, even if a matrix has only real entries, both {@code U} and {@code T} may contain complex values.</p>
 *
 * @param <T> The type of matrix to compute the Schur decomposition of.
 * @param <U> Vector type for columns of the matrices of the Schur decomposition.
 */
public abstract class SchurDecomposition<
        T extends MatrixMixin<T, ?, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, ?, ?, ?, ?, T, ?, ?>>
        implements Decomposition<T> {

    /**
     * Tolerance for considering an element zero.
     */
    static protected final double TOL = Math.ulp(1.0d);

    /**
     * The minimum number of iterations to perform by default in the QR algorithm if it has not converged. Specifically, the
     * QR algorithm will run for max(MIN_ITERATIONS, maxIterations) iteration or until the QR algorithm converges,
     * whichever comes first.
     */
    protected final static int MIN_DEFAULT_ITERATIONS = 500;

    /**
     * The number of iterations to run before performing an exceptional shift.
     */
    protected final static int EXCEPTIONAL_SHIT_TOL = 30;

    /**
     * The maximum number of iterations to run the QR algorithm for.
     */
    protected final static int MAX_ITERATIONS = 25*EXCEPTIONAL_SHIT_TOL - 1;

    /**
     * Tracks the number of exceptional shifts.
     */
    protected int numExceptionalShifts;

    /**
     * Flag which indicates if the unitary {@code U} should be computed in the decomposition.
     */
    protected final boolean computeU;

    /**
     * Storage for the unitary {@code U} matrix in the Schur decomposition corresponding to {@code A=UTU<sup>H</sup>}.
     */
    protected CMatrix U;

    /**
     * Storage for the upper triangular {@code T} matrix in the Schur decomposition corresponding to
     * {@code A=UTU<sup>H</sup>}.
     */
    protected CMatrix T;

    /**
     * Decomposer to compute the Hessenburg matrix similar to the source matrix. This Hessenburg
     * matrix will be the actual matrix that the Schur decomposition is computed for.
     */
    protected HessenburgDecomposition<T, U> hess;


    /**
     * Creates a decomposer to compute the Schur decomposition.
     */
    protected SchurDecomposition() {
        computeU = true;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition which will run for at most {@code maxIterations}
     * iterations.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    protected SchurDecomposition(boolean computeU) {
        this.computeU = computeU;
    }


    /**
     * Computes the Schur decomposition of a matrix using Francis' double shift
     * algorithm (i.e. The implicit double shifted QR algorithm).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    protected abstract void shiftedImplicitQR(T H);


    /**
     * Computes the Schur decomposition of the {@code src} matrix using a variant of the QR algorithm.
     * @param src The matrix to decompose.
     */
    protected final void applyQR(T src) {
        ParameterChecks.assertSquare(src.shape());
        hess.decompose(src); // Compute a Hessenburg matrix which is similar to src (i.e. has the same eigenvalues).

        shiftedImplicitQR(hess.getH()); // Use shifted implicit QR algorithm.
    }


    /**
     * Gets the upper triangular matrix {@code T} from the Schur decomposition corresponding to {@code A=UTU<sup>H</sup>}.
     * @return The {@code T} from the Schur decomposition corresponding to {@code A=UTU<sup>H</sup>}.
     */
    public CMatrix getT() {
        return T;
    }


    /**
     * Gets the unitary matrix {@code U} from the Schur decomposition corresponding to {@code A=UTU<sup>H</sup>}.
     * @return The {@code U} from the Schur decomposition corresponding to {@code A=UTU<sup>H</sup>}.
     */
    public CMatrix getU() {
        return U;
    }


    /**
     * Converts a matrix decomposed into a real Schur form to a complex Schur form. Note, this
     * will not update the {@code U} matrix from the Schur decomposition. See {@link #real2ComplexSchur(Matrix, Matrix)}
     * to also update the {@code U} matrix.
     * @param realT The real Schur matrix {@code T} in the Schur decomposition {@code {@code A=UTU<sup>T</sup>}}
     * @return A matrix in complex Schur from corresponding to {@code realT}.
     * @see #real2ComplexSchur(Matrix, Matrix)
     */
    // Code adapted from scipy rsf2csf https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rsf2csf.html
    public static CMatrix real2ComplexSchur(Matrix realT) {
        return real2ComplexSchur(realT, null)[0];
    }


    /**
     * Converts a matrix decomposed into a real Schur form to a complex Schur form.
     * @param realT The real Schur matrix {@code T} in the Schur decomposition {@code {@code A=UTU<sup>T</sup>}}
     * @param realU The orthogonal matrix {@code U} in the Schur decomposition {@code A=UTU<sup>T</sup>}. Can be null.
     * @return An array of complex matrices corresponding to the complex Schur form of the two matrices {@code realT} and
     * {@code realU} in order.
     * @see #real2ComplexSchur(Matrix)
     */
    // Code adapted from scipy rsf2csf https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rsf2csf.html
    public static CMatrix[] real2ComplexSchur(Matrix realT, Matrix realU) {
        CMatrix T = realT.toComplex();
        CMatrix U = realU == null ? null : realU.toComplex();

        for(int m=realT.numRows-1; m>0; m--) {

            // Check for convergence.
            if(notConverged(T, m)) {
                // Then a 2x2 block must be deflated.
                deflateT(T, U, m);
            } else {
                T.set(0, m, m-1); // Ensure the sub-diagonal value is set to zero.
            }
        }

        return new CMatrix[]{T, U};
    }


    /**
     * Deflates a 2x2 block of the {@code T} matrix from the Schur decomposition and accumulates similarity transformations
     * used in deflation in the {@code U} matrix of the Schur decomposition.
     * @param T The {@code T} matrix from the Schur decomposition.
     * @param U The {@code U} matrix of the Schur decomposition.
     * @param m Row and column index of the lower right entry of the 2x2 block to deflate in {@code T}.
     */
    // Code adapted from scipy rsf2csf https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rsf2csf.html
    static void deflateT(CMatrix T, CMatrix U, int m) {
        // Compute the eigenvalues of the 2x2 block.
        CVector mu = Eigen.get2x2EigenValues(
                T.getSlice(m-1, m+1, m-1, m+1)
        ).sub(T.get(m, m));

        CMatrix G = Givens.get2x2Rotator(new CVector(mu.get(0), new CNumber(T.get(m, m-1))));

        // Apply rotation to T matrix to bring it into upper triangular form
        T.setSlice(
                G.mult(T.getSlice(m-1, m+1, m-1, T.numCols)),
                m-1, m-1
        );
        T.setSlice(
                T.getSlice(0, m+1, m-1, m+1).mult(G.H()),
                0, m-1
        );

        if(U != null) {
            // Accumulate similarity transforms in the U matrix.
            U.setSlice(
                    U.getSlice(0, U.numRows, m-1, m+1).mult(G.H()),
                    0, m-1
            );
        }

        T.set(0, m, m-1);
    }


    /**
     * Checks if an entry along the first sub-diagonal of the {@code T} matrix in the Schur decomposition has
     * converged to zero within machine precision. A value is considered to be converged if it is small relative to the
     * two values on the diagonal of the {@code T} matrix immediately next to the value of interest. That is, a
     * value at index {@code (m, m-1)} is considered converged if it is less in absolute value than the absolute
     * sum of the entries at {@code (m-1, m-1)} and {@code (m, m)} times machine epsilon
     * (i.e. {@link Math#ulp(double)  Math.ulp(1.0d)}).
     * @param T The {@code T} Matrix in the Schur decomposition.
     * @param m Row index of the value of interest within the {@code T} matrix.
     * @return True if the specified entry has not converged. That is, the entry in the {@code T} matrix is greater
     * than (in absolute value) machine precision (i.e. Math.ulp(1.0)) times the absolute sum of the entries along the
     * block 2x2 matrix on the diagonal of {@code T} containing the entry. Otherwise, returns false.
     */
    protected static boolean notConverged(CMatrix T, int m) {
        return T.get(m, m-1).mag() > TOL*( T.get(m-1, m-1).mag() + T.get(m, m).mag() );
    }


    /**
     * Checks if an entry along the first sub-diagonal of the {@code T} matrix in the Schur decomposition has
     * converged to zero within machine precision. A value is considered to be converged if it is small relative to the
     * two values on the diagonal of the {@code T} matrix immediately next to the value of interest. That is, a
     * value at index {@code (m, m-1)} is considered converged if it is less in absolute value than the absolute
     * sum of the entries at {@code (m-1, m-1)} and {@code (m, m)} times machine epsilon
     * (i.e. {@link Math#ulp(double)  Math.ulp(1.0d)}).
     * @param T The {@code T} Matrix in the Schur decomposition.
     * @param m Row index of the value of interest within the {@code T} matrix.
     * @return True if the specified entry has not converged. That is, the entry in the {@code T} matrix is greater
     * than (in absolute value) machine precision (i.e. Math.ulp(1.0)) times the absolute sum of the entries along the
     * block 2x2 matrix on the diagonal of {@code T} containing the entry. Otherwise, returns false.
     */
    protected static boolean notConverged(Matrix T, int m) {
        return Math.abs(T.get(m, m-1)) > TOL*Math.abs( T.get(m-1, m-1) + Math.abs(T.get(m, m)) );
    }
}
