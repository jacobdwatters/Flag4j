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
import com.flag4j.linalg.Eigen;
import com.flag4j.linalg.transformations.Householder;

/**
 * <p>This class computes the Schur decomposition of a square matrix.
 * That is, decompose a square matrix {@code A} into {@code A=QUQ<sup>H</sup>} where {@code Q} is a unitary
 * matrix whose columns are the eigenvectors of {@code A} and {@code U} is an upper triangular matrix in
 * Schur form whose diagonal entries are the eigenvalues of {@code A}, corresponding to the columns of {@code Q},
 * repeated per their multiplicity.</p>
 *
 * <p>Note, even if a matrix has only real entries, both {@code Q} and {@code U} may contain complex values.</p>
 */
public class ComplexSchurDecomposition extends SchurDecomposition<CMatrix, CVector> {


    /**
     * Creates a Schur decomposer for a real dense matrix.
     */
    public ComplexSchurDecomposition() {
        super();
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new ComplexHessenburgDecomposition(computeU);
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    public ComplexSchurDecomposition(boolean computeU) {
        super(computeU);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new ComplexHessenburgDecomposition(computeU);
    }


    /**
     * <p>Computes the Schur decomposition for a real dense square matrix.
     * That is, decompose a square matrix {@code A} into {@code A=QUQ<sup>H</sup>} where {@code Q} is a unitary
     * matrix whose columns are the eigenvectors of {@code A} and {@code U} is an upper triangular matrix in
     * Schur form whose diagonal entries are the eigenvalues of {@code A}, corresponding to the columns of {@code Q},
     * repeated per their multiplicity.</p>
     *
     * <p>Note, even if a matrix has only real entries, both {@code Q} and {@code U} may contain complex values.</p>
     *
     * @param src The source matrix to compute the Schur decomposition of.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexSchurDecomposition decompose(CMatrix src) {
        applyQR(src); // Apply a variant of the QR algorithm.

        if(computeU) {
            // Collect Hessenburg decomposition similarity transformations.
            U = hess.Q.mult(U);
        }

        return this;
    }


    /**
     * Computes the Schur decomposition for a complex matrix using a shifted QR algorithm where the QR decomposition
     * is computed explicitly.
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    @Override
    protected void shiftedExplicitQR(CMatrix H) {
        final int maxIterations = Math.max(10*H.numRows, MIN_DEFAULT_ITERATIONS);

        QRDecomposition<CMatrix, CVector> qr = new ComplexQRDecomposition();
        final double tol = Math.ulp(1.0d); // Tolerance for considering an entry zero.

        int iters; // Number of iterations.

        // Initialize matrices.
        T = H;
        U = computeU ? CMatrix.I(T.numRows) : null;

        int m = T.numRows-1;

        while(m > 0) {
            iters = 0; // Reset the number of iterations.

            while(notConverged(T, m) && iters < maxIterations) {
                iters++;

                CNumber shift = getWilkinsonShift(T); // Compute shift.
                CMatrix mu = Matrix.I(T.numRows).mult(shift);

                qr.decompose(T.sub(mu));

                T = qr.R.mult(qr.Q).add(mu);
            }

            if(iters<maxIterations) {
                // Deflate 1x1 block
                m--;

            } else {
                // Deflate 2x2 block
                deflateT(T, U, m);
                m-=2;
            }
        }
    }


    /**
     * Computes the Wilkinson shift for a complex matrix. That is, the eigenvalue of the lower left 2x2 sub matrix
     * which is closest in magnitude to the lower left entry of the matrix.
     * @param src Matrix to compute the Wilkinson shift for.
     * @return The Wilkinson shift for the {@code src} matrix.
     */
    private CNumber getWilkinsonShift(CMatrix src) {
        // Compute eigenvalues of lower 2x2 block.
        CVector lambdas = Eigen.get2x2LowerRightBlockEigenValues(src);
        CNumber v = src.entries[src.entries.length-1];

        if(v.sub(lambdas.entries[0]).mag() < v.sub(lambdas.entries[1]).mag()) {
            return lambdas.entries[0];
        } else {
            return lambdas.entries[1];
        }
    }


    /**
     * Computes the Schur decomposition of a matrix using Francis' algorithm (i.e. The implicit double shifted
     * QR algorithm).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    @Override
    protected void doubleShiftImplicitQR(CMatrix H) {
        final int maxIterations = Math.max(10*H.numRows, MIN_DEFAULT_ITERATIONS);
        int iters;

        HessenburgDecomposition<CMatrix, CVector> hess = new ComplexHessenburgDecomposition(computeU);

        // Initialize matrices.
        T = H;
        U = computeU ? CMatrix.I(T.numRows) : null;
        int m = T.numRows-1;

        while(m > 0) {
            iters = 0; // Reset the number of iterations.

            while(notConverged(T, m) && iters < maxIterations) {
                iters++;

                // Compute eigenvalues of lower right 2x2 block.
                CVector rho = Eigen.get2x2LowerRightBlockEigenValues(T);

                CVector p = new CVector(
                        T.get(0, 0).sub(rho.get(0)).mult( T.get(0, 0).sub(rho.get(1)) ).add( T.get(0, 1).mult(T.get(1, 0)) ),
                        T.get(1, 0).mult( T.get(0, 0).add(T.get(1, 1)).sub(rho.get(0)).sub(rho.get(1)) ),
                        T.get(2, 1).mult(T.get(1, 0))
                );

                CMatrix ref = Householder.getReflector(p);

                // Apply transform and create bulge.
                T.setSlice(
                        ref.H().mult(T.getSlice(0, 3, 0, T.numCols)),
                        0, 0
                );
                T.setSlice(
                        T.getSlice(0, T.numRows, 0, 3).mult(ref),
                        0, 0
                );

                T = hess.decompose(T).getH(); // A crude bulge chase using the Hessenburg decomposition.

                if(computeU) {
                    // Collect similarity transformations in U matrix.
                    U.setSlice(
                            U.getSlice(0, U.numRows, 0, 3).mult(ref),
                            0, 0
                    );

                    U = U.mult(hess.getQ());
                }

            }

            if(iters<maxIterations) {
                // Deflate 1x1 block.
                T.set(0, m, m-1); // Ensure the sub-diagonal value is set to zero.
                m--;
            } else {
                // Deflate 2x2 block.
                deflateT(T, U, m);
                m-=2;
            }
        }
    }
}
