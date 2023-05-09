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
import com.flag4j.core.MatrixMixin;
import com.flag4j.linalg.Eigen;
import com.flag4j.linalg.transformations.Householder;

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
 */
public abstract class SchurDecomposition<T extends MatrixMixin<T, ?, ?, ?, ?, ?>> implements Decomposition<T> {

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
     * Computes the Schur decomposition for a complex dense matrix using a double shifted implicit QR algorithm
     * (also known as Francis's Algorithm of degree two).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    // TODO: This is a very rudimentary implementation of Francis's double shift implicit QR algorithm.
    //  Several improvements must be made. See Fundamentals of matrix computations Vol. 3 for more information.
    //  this may also only work for real symmetric matrices...
    protected void doubleShiftImplicitQR(CMatrix H) {
        if(useDefaultMaxIterations) {
            // The algorithm should converge within machine precision in O(n^3).
            maxIterations = (int) Math.max(Math.pow(H.numRows, 3), MIN_DEFAULT_ITERATIONS);
        }

        ComplexHessenburgDecomposition hess = new ComplexHessenburgDecomposition(computeU); // Would need to compute Q to compute eigenvectors as well.

        T = H;

        if(computeU) {
            U = CMatrix.I(T.numRows);
        }

        for(int i=0; i<maxIterations; i++) {
            // Compute shifts as eigenvalues of lower right 2x2 block.
            // TODO: Ensure this is numerically stable. Also, add random shifts as noted in Fundamentals of Matrix Computations Vol. 3.
            CVector rho = Eigen.get2x2LowerLeftBlockEigenValues(T);

            CNumber[] pEntries = {
                    T.entries[0].sub(rho.entries[0]).mult(T.entries[0].sub(rho.entries[1])).add(T.entries[1].mult(T.entries[T.numCols])),
                    T.entries[T.numCols].mult( T.entries[0].add(T.entries[T.numCols + 1]).sub(rho.entries[0]).sub(rho.entries[1]) ),
                    T.entries[2*T.numCols+1].mult(T.entries[T.numCols])
            };
            Vector p = new CVector(pEntries).toReal(); // Should be real anyway
            Matrix Q = Householder.getReflector(p);

            // Apply the Householder reflector to T.
            T.setSlice(Q.T().mult(T.getSlice(0, Q.numRows, 0, T.numCols)), 0, 0);
            T.setSlice(T.getSlice(0, T.numRows, 0, Q.numCols).mult(Q), 0, 0);

            if(computeU) {
                // Apply the Householder reflector to U.
                U.setSlice(U.getSlice(0, U.numRows, 0, Q.numCols).mult(Q), 0, 0);
            }

            // TODO: A crude bulge chase using Hessenberg decomposition. Since we know A is nearly in Hessenberg already, this
            //  will perform a lot of superfluous computations. Use Householder transformations instead.
            T = hess.decompose(T).getH();

            if(computeU) {
                U = U.mult(hess.getQ());
            }

            if(T.roundToZero().isTriU()) {
                break; // We have converged (approximately) to an upper triangular matrix.
            }
        }
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
