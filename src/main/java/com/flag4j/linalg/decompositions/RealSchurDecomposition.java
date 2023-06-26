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
import com.flag4j.linalg.Eigen;
import com.flag4j.linalg.transformations.Householder;


/**
 * <p>This class computes the Schur decomposition of a square matrix.
 * That is, decompose a square matrix {@code A} into {@code A=UTU<sup>H</sup>} where {@code U} is a unitary
 * matrix and {@code T} is an upper triangular matrix (or possibly block
 * upper triangular matrix) in Schur form whose diagonal entries are the eigenvalues of {@code A}.</p>
 *
 * <p>Note, even if a matrix has only real entries, both {@code Q} and {@code U} may contain complex values.</p>
 */
public class RealSchurDecomposition extends SchurDecomposition<Matrix, Vector> {

    /**
     * Flag which indicates if {@code T} should be stored in real or complex Schur form.
     * If true, {@code T} will be stored in real schur form (default). If false, {@code T}\
     * will be stored in complex schur form.
     */
    private final boolean realSchur;

    /**
     * Storage for real Schur form of the {@code T} matrix.
     */
    private Matrix realT;
    /**
     * Storage for real Schur form of the {@code U} matrix.
     */
    private Matrix realU;

    /**
     * Creates a Schur decomposer for a real dense matrix. {@code T} will be stored in
     * Real Schur form. To convert {@code T} to real schur from see
     * {@link SchurDecomposition#real2ComplexSchur(Matrix, Matrix)} or
     */
    public RealSchurDecomposition() {
        super();
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(super.computeU);
        realSchur = true;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    public RealSchurDecomposition(boolean computeU) {
        super(computeU);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(computeU);
        realSchur = true;
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     * @param realSchur Flag which indicates weather the real schur form or complex schur form should be computed.
     */
    public RealSchurDecomposition(boolean computeU, boolean realSchur) {
        super(computeU);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(computeU);
        this.realSchur = realSchur;
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
    public RealSchurDecomposition decompose(Matrix src) {
        applyQR(src); // Apply a variant of the QR algorithm.

        if(computeU) {
            // Collect Hessenburg decomposition similarity transformations.
            realU = hess.Q.mult(realU);
        }

        if(realSchur) {
            T = realT.toComplex();
            U = realU.toComplex();
        } else {
            // Convert to the complex schur form.
            CMatrix[] TU = SchurDecomposition.real2ComplexSchur(realT, realU);
            T = TU[0];
            U = TU[1];
        }

        return this;
    }


    /**
     * Computes the Schur decomposition for a complex matrix using a shifted QR algorithm where the QR decomposition
     * is computed explicitly.
     * @param H - The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    @Override
    protected void shiftedExplicitQR(Matrix H) {
        final int maxIterations = Math.max(10*H.numRows, MIN_DEFAULT_ITERATIONS);
        final double tol = Math.ulp(1.0d); // Tolerance for considering an entry zero.
        int iters; // Number of iterations.

        QRDecomposition<Matrix, Vector> qr = new RealQRDecomposition();

        // Initialize matrices.
        realT = H;
        realU = computeU ? Matrix.I(realT.numRows) : null;

        int m = realT.numRows-1;

        double shift = realT.entries[realT.entries.length-1]; // Compute shift.

        while(m > 0) {
            iters = 0; // Reset the number of iterations.

            while(realT.getSlice(m, m+1, 0, m).maxAbs() > tol && iters < maxIterations) {
                iters++;

                Matrix mu = Matrix.I(realT.numRows).mult(shift);
                qr.decompose(realT.sub(mu));
                realT = qr.R.mult(qr.Q).add(mu);

                if(computeU) {
                    realU = realU.mult(qr.Q);
                }
            }

            if(iters<maxIterations) {
                realT.set(0, m, m-1); // Ensure the sub-diagonal value is set to zero.
                m--; // 1x1 block.
            } else {
                m-=2; // 2x2 block
            }
        }
    }


    /**
     * Computes the Schur decomposition of a matrix using Francis' algorithm (i.e. The implicit double shifted
     * QR algorithm).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    @Override
    protected void doubleShiftImplicitQR(Matrix H) {
        final int maxIterations = Math.max(10*H.numRows, MIN_DEFAULT_ITERATIONS);
        int iters;

        HessenburgDecomposition<Matrix, Vector> hess = new RealHessenburgDecomposition(computeU);

        // Initialize matrices.
        realT = H;
        realU = computeU ? Matrix.I(realT.numRows) : null;
        
        int m = realT.numRows-1;

        while(m > 0) {
            iters = 0; // Reset the number of iterations.

            while(notConverged(realT, m) && iters < maxIterations) {
                iters++;

                // Compute eigenvalues of lower right 2x2 block.
                CVector rho = Eigen.get2x2LowerRightBlockEigenValues(realT);

                CNumber a = new CNumber(realT.get(0, 0));
                CNumber b = new CNumber(realT.get(0, 1));
                CNumber c = new CNumber(realT.get(1, 0));
                CNumber d = new CNumber(realT.get(2, 1));

                Vector p = new CVector(
                        a.sub(rho.get(0)).mult( a.sub(rho.get(1)) ).add( b.mult(realT.get(1, 0)) ),
                        c.mult( a.add(realT.get(1, 1)).sub(rho.get(0)).sub(rho.get(1)) ),
                        d.mult(realT.get(1, 0))
                ).toReal(); // Should be real.

                Matrix ref = Householder.getReflector(p);

                // Apply transform and create bulge.
                realT.setSlice(
                        ref.H().mult(realT.getSlice(0, 3, 0, realT.numCols)),
                        0, 0
                );
                realT.setSlice(
                        realT.getSlice(0, realT.numRows, 0, 3).mult(ref),
                        0, 0
                );

                realT = hess.decompose(realT).getH(); // A crude bulge chase using the Hessenburg decomposition.

                if(computeU) {
                    // Collect similarity transformations in realU matrix.
                    realU.setSlice(
                            realU.getSlice(0, realU.numRows, 0, 3).mult(ref),
                            0, 0
                    );

                    realU = realU.mult(hess.getQ());
                }

            }

            if(iters<maxIterations) {
                // Deflate 1x1 block.
                realT.set(0, m, m-1); // Ensure the sub-diagonal value is set to zero.
                m--;
            } else {
                // Deflate 2x2 block.
                m-=2;
            }
        }
    }


    /**
     * Gets the real Schur form of the {@code T} matrix from the Schur decomposition.
     * @return The real Schur form of the {@code T} matrix from the Schur decomposition.
     */
    public Matrix getRealT() {
        return realT;
    }


    /**
     * Gets the {@code U} matrix in the Schur decomposition corresponding to the real
     * Schur form of the {@code T} matrix in the Schur decomposition.
     * @return The{@code U} matrix in the Schur decomposition corresponding to the real
     * Schur form of the {@code T} matrix in the Schur decomposition.
     */
    public Matrix getRealU() {
        return realU;
    }
}
