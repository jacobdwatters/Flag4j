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
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.MatrixBase;

import java.util.Scanner;

/**
 * <p>This abstract class specifies methods for computing the Schur decomposition of a square matrix.
 * That is, decompose a square matrix {@code A} into {@code A=QUQ<sup>H</sup>} where {@code Q} is a unitary
 * matrix whose columns are the eigenvectors of {@code A} and {@code U} is an upper triangular matrix in
 * Schur form whose diagonal entries are the eigenvalues of {@code A}, corresponding to the columns of {@code Q},
 * repeated per their multiplicity.</p>
 *
 * <p>Note, even if a matrix has only real entries, both {@code Q} and {@code U} may contain complex values.</p>
 *
 * @param <T> The type of matrix to compute the Schur decomposition of.
 */
public abstract class SchurDecomposition<T extends MatrixBase<?>> implements Decomposition<T> {

    /**
     * The maximum number of iterations to run the {@code QR} algorithm when computing the Schur decomposition. This
     * overrides the minimum number of iterations.
     */
    protected int maxIterations;
    /**
     * The minimum number of iterations to perform by default in the QR algorithm if it has not converged. Specifically, the
     * QR algorithm will run for max(MIN_ITERATIONS, maxIterations) iteration or until the QR algorithm converges,
     * whichever comes first.
     */
    protected final static int MIN_DEFAULT_ITERATIONS = 500;
    /**
     * Flag for indicating if the default number of max iteration should be used in the QR algorithm.
     * If true, the max iterations will be set to the number of rows cubed. If false, the max iterations is
     * specified by the user.
     */
    protected final boolean useDefaultMaxIterations;
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
    protected HessenburgDecomposition<T> hess;

    public boolean debug = false; // TODO: Temporary for testing. Must be removed.


    /**
     * Creates a decomposer to compute the Schur decomposition.
     */
    protected SchurDecomposition() {
        maxIterations = -1;
        useDefaultMaxIterations = true;
        computeU = true;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition which will run for at most {@code maxIterations}
     * iterations.
     * @param maxIterations Maximum number of iterations to run when computing the Schur decomposition.
     */
    protected SchurDecomposition(int maxIterations) {
        this.maxIterations = maxIterations;
        useDefaultMaxIterations = false;
        computeU = true;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition which will run for at most {@code maxIterations}
     * iterations.
     * @param maxIterations Maximum number of iterations to run when computing the Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    protected SchurDecomposition(boolean computeU, int maxIterations) {
        this.maxIterations = maxIterations;
        this.computeU = computeU;
        useDefaultMaxIterations = false;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    protected SchurDecomposition(boolean computeU) {
        maxIterations = -1;
        this.computeU = computeU;
        useDefaultMaxIterations = true;
    }


    /**
     * Computes the Schur decomposition for a complex dense matrix.
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    protected void computeSchurDecomp(CMatrix H) {
        if(useDefaultMaxIterations) {
            // The algorithm should converge within machine precision in O(n^3) in general.
            maxIterations = (int) Math.max(Math.pow(H.numRows, 3), MIN_DEFAULT_ITERATIONS);
        }

        double tol = 1e-14;
        int count;
        int n = H.numRows-1;

        T = CMatrix.I(H.numRows);
        U = CMatrix.I(H.numRows);

        ComplexQRDecomposition qr = new ComplexQRDecomposition(); // Decomposer for use in the QR algorithm.
        CMatrix Q; // Q matrix from QR decomposition.
        CMatrix R; // R matrix from QR decomposition.

        CMatrix mu;

        CNumber disc;
        CVector lam = new CVector(H.numRows);

        while(n>0) {
            count = 0;

            // Apply the QR algorithm (Shifted QR algorithm using Rayleigh shift).
            while(H.getSlice(n, n+1, 0, n).maxAbs() > tol && count<maxIterations) {
                count++;
                mu = CMatrix.I(n+1).mult(H.entries[n*(H.numCols + 1)]); // Construct diagonal matrix.

                qr.decompose(H.sub(mu)); // Compute the QR decomposition with a shift.
                Q = qr.getQ();
                R = qr.getR();

                H = R.mult(Q).add(mu); // Reverse the shift.

                if(computeU) {
                    Q = CMatrix.I(U.numRows).setSliceCopy(Q, 1, 1);
                    U = Q.mult(U);
                }
            }

            T.setSlice(H, H.numRows-(n+1), H.numRows-(n+1));

            if(count<maxIterations) {
                // Then there is an isolated 1x1 block.
//                lam.entries[n] = H.entries[n*(H.numCols + 1)];
                H = H.getSlice(0, n, 0, n);
                n--; // Deflate H by 1.

            } else {
                System.out.println("Here");
                // Then there is an isolated 2x2 block.
//                disc = CNumber.pow(H.entries[(n-1)*(H.numCols + 1)].sub(H.entries[n*(H.numCols + 1)]), 2);
//                disc.addEq(H.entries[n*(H.numCols + 1) - 1].mult(H.entries[(n-1)*H.numCols + n]).mult(4));
//
//                lam.entries[n] = (H.entries[(n-1)*(H.numCols + 1)].add(H.entries[n*(H.numCols + 1)]).add(CNumber.sqrt(disc))).div(2.0);
//                lam.entries[n-1] = (H.entries[(n-1)*(H.numCols + 1)].add(H.entries[n*(H.numCols + 1)]).sub(CNumber.sqrt(disc))).div(2.0);

//                T.setSlice(new CMatrix(2), n-1, n-1);
//                T.entries[n*(H.numCols + 1)] = lam.entries[n];
//                T.entries[(n-1)*(H.numCols + 1)] = lam.entries[n-1];

                n-=2; // Deflate H by 2.
                H = H.getSlice(0, n+1, 0, n+1);
            }
        }
    }


    /**
     * Computes the Schur decomposition for a complex dense matrix using the basic shifted QR algorithm (Rayleigh shift).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    protected void shiftedQR(CMatrix H) {
        if(useDefaultMaxIterations) {
            // The algorithm should converge within machine precision in O(n^3).
            maxIterations = (int) Math.max(Math.pow(H.numRows, 3), MIN_DEFAULT_ITERATIONS);
        }

        int count = 0;
        int n = H.numRows-1;

        T = H;
        U = CMatrix.I(H.numRows);

        ComplexQRDecomposition qr = new ComplexQRDecomposition(); // Decomposer for use in the QR algorithm.
        CMatrix Q; // Q matrix from QR decomposition.
        CMatrix R; // R matrix from QR decomposition.

        CMatrix mu;

        // Apply the QR algorithm (Shifted QR algorithm using Rayleigh shift).
        while(count<maxIterations) {
            count++;
//            mu = CMatrix.I(n+1).mult(T.entries[T.entries.length-1]); // Construct diagonal matrix.
            mu = CMatrix.I(n+1).mult(CNumber.exp(CNumber.IMAGINARY_UNIT)); // Construct diagonal matrix.

            qr.decompose(T.sub(mu)); // Compute the QR decomposition with a shift.
            Q = qr.getQ();
            R = qr.getR();

            T = R.mult(Q).add(mu); // Reverse the shift.
            U = U.mult(Q);
        }
    }


    /**
     * Computes the Schur decomposition for a complex dense matrix using a double shifted implicit QR algorithm
     * (also known as Francis's Algorithm of degree two).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    protected void doubleShiftImplicitQR(CMatrix H) {
        // NOTE: This is a quick and dirty implementation and should only be used as reference.
        //  See Fundamentals of Matrix Computations (Watkins) p 368-369.
        // Right now, this only computes eigenvalues which are represented in the real Schur form.
        if(useDefaultMaxIterations) {
            // The algorithm should converge within machine precision in O(n^3).
            maxIterations = (int) Math.max(Math.pow(H.numRows, 3), MIN_DEFAULT_ITERATIONS);
        }

        RealQRDecomposition qr = new RealQRDecomposition();
        ComplexHessenburgDecomposition hess = new ComplexHessenburgDecomposition(computeU); // Would need to compute Q to compute eigenvectors as well.
        T = H;
        if(computeU) {
            U = CMatrix.I(T.numRows);
        }

        Scanner stdin =  new Scanner(System.in);

        for(int i=0; i<maxIterations; i++) {
            // Compute shifts as eigenvalues of lower right 2x2 block.
            CVector rho = get2x2BlockEigenValues(T);

            if(debug) {
                System.out.println("Iteration: " + i + "\n" + "-".repeat(100) + "\n");
                System.out.println("T_start:\n" + T + "\n");
            }

            CNumber[] pEntries = {
                    T.entries[0].sub(rho.entries[0]).mult(T.entries[0].sub(rho.entries[1])).add(T.entries[1].mult(T.entries[T.numCols])),
                    T.entries[T.numCols].mult( T.entries[0].add(T.entries[T.numCols + 1]).sub(rho.entries[0]).sub(rho.entries[1]) ),
                    T.entries[2*T.numCols+1].mult(T.entries[T.numCols])
            };
            Vector p = new CVector(pEntries).toReal(); // Should be real anyway
            qr.decompose(p.toMatrix()); // Easy way to build an orthogonal matrix with first column proportional to p.
            Matrix Q = qr.getQ(); // Realistically, the QR decomposition need not be computed explicitly.

            T.setSlice(Q.T().mult(T.getSlice(0, 3, 0, T.numCols)), 0, 0);
            T.setSlice(T.getSlice(0, 4, 0, 3).mult(Q), 0, 0); // Create bulge.

            if(debug) {
                System.out.println("T_after_transforms:\n" + T + "\n");
            }

            // A crude bulge chase using Hessenberg decomposition. Since we know A is nearly in Hessenberg already, this
            // will perform a lot of superfluous computations.
            T = hess.decompose(T).getH();

            if(computeU) {
                U = hess.Q.mult(U);
            }

            if(debug) {
                System.out.println("T_hess:\n" + T);
                stdin.nextLine();
                System.out.println("-".repeat(100) + "\n\n\n");
            }

            if(T.roundToZero().isTriU()) {
                break; // We have converged (approximately) to an upper triangular matrix.
            }
        }
    }


    /**
     * Computes the eigenvalues for the lower right 2x2 block matrix with a larger matrix.
     * @param src Source matrix to compute eigenvalues of lower right 2x2 block.
     * @return A vector of length 2 containing the eigenvalues of the lower right 2x2 block of {@code src}.
     */
    private CVector get2x2BlockEigenValues(CMatrix src) {
        CVector shifts = new CVector(2);
        int n = src.numRows-1;

        // Get the four entries from lower right 2x2 sub-matrix.
        CNumber a = src.entries[(n-1)*(src.numCols + 1)];
        CNumber b = src.entries[(n-1)*src.numCols + n];
        CNumber c = src.entries[n*(src.numCols + 1) - 1];
        CNumber d = src.entries[(n)*(src.numCols + 1)];

        CNumber det = a.mult(d).sub(b.mult(c)); // 2x2 determinant.
        CNumber htr = a.add(b).div(2); // Half of the 2x2 trace.

        // 2x2 block eigenvalues.
        shifts.entries[0] = htr.add(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));
        shifts.entries[1] = htr.sub(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));

        return shifts;
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
}
