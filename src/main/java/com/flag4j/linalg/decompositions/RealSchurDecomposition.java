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
import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.util.ParameterChecks;



/**
 * <p>This class computes the Schur decomposition of a square matrix.
 * That is, decompose a square matrix {@code A} into {@code A=UTU<sup>H</sup>} where {@code U} is a unitary
 * matrix and {@code T} is an upper triangular matrix (or possibly block
 * upper triangular matrix) in Schur form whose diagonal entries are the eigenvalues of {@code A}.</p>
 *
 * <p>Note, even if a matrix has only real entries, both {@code Q} and {@code U} may contain complex values.</p>
 */
public class RealSchurDecomposition extends SchurDecomposition<Matrix, Vector> {

    private final boolean realSchur;
    private Matrix realT;
    private Matrix realU;

    /**
     * Creates a Schur decomposer for a real dense matrix.
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
        ParameterChecks.assertSquare(src.shape);
        hess.decompose(src); // Compute a Hessenburg matrix which is similar to src (i.e. has the same eigenvalues).g

        // Compute the real schur form.
        shiftedExplicitRealQR(hess.getH());

        if(!realSchur) {
            // Convert to the complex schur form.
            CMatrix[] TU = SchurDecomposition.real2ComplexSchur(realT, realU);
            T = TU[0];
            U = TU[1];
        } else {
            T = realT.toComplex();
            U = realU.toComplex();
        }

        if(computeU) {
            // Convert Hessenburg eigenvectors to the eigenvectors of the source matrix.
            U = hess.Q.mult(U);
        }

        return this;
    }


    /**
     * Computes the Schur decomposition for a complex matrix using a shifted QR algorithm where the QR decomposition
     * is computed explicitly.
     * @param H - The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    protected void shiftedExplicitRealQR(Matrix H) {
        final int maxIterations = Math.max(10*H.numRows, MIN_DEFAULT_ITERATIONS);
        final double tol = Math.ulp(1.0d); // Tolerance for considering an entry zero.
        int iters; // Number of iterations.

        QRDecomposition<Matrix, Vector> qr = new RealQRDecomposition();

        // Initialize matrices.
        realT = H;
        realU = computeU ? Matrix.I(realT.numRows) : null;

        int n = realT.numRows;

        double shift = realT.entries[realT.entries.length-1]; // Compute shift.

        while(n > 1) {
            iters = 0; // Reset the number of iterations.

            while(realT.getSlice(n-1, n, 0, n-1).maxAbs() > tol && iters < maxIterations) {
                iters++;

                Matrix mu = Matrix.I(realT.numRows).mult(shift);
                qr.decompose(realT.sub(mu));
                realT = qr.R.mult(qr.Q).add(mu);

                if(computeU) {
                    realU = realU.mult(qr.Q);
                }
            }

            if(iters < maxIterations) {
                n--; // 1x1 block.
            } else {
                n-=2; // 2x2 block
            }
        }
    }


    /**
     * Gets the real Schur form of the {@code T} matrix from the Schur decomposition
     * @return
     */
    public Matrix getRealT() {
        return realT;
    }


    public Matrix getComplexT() {
        return realT;
    }


    public static void main(String[] args) {
//        PrintOptions.setPrecision(50);

        RealSchurDecomposition schur = new RealSchurDecomposition(true, false);

        double[][] aEntries = {
                {3.45, -99.34, 14.5, 24.5},
                {-0.0024, 0, 25.1, 1.5},
                {100.4, 5.6, -4.1, -0.002}
        };
        Matrix A = new Matrix(aEntries);
        A = A.invDirectSum(A.H());

        double[][] bEntries = {
                {1,  1, 1, 1, 1},
                {-1, 1, 1, 1, 1},
                {0, -1, 1, 1, 1},
                {0, 0, -1, 1, 1},
                {0, 0, 0, -1, 1}
        };
        Matrix B = new Matrix(bEntries);

        Matrix src = A;

        schur.decompose(src);
        CMatrix U = schur.getU();
        CMatrix T = schur.getT();

        System.out.println("Results:\n" + "-".repeat(80));
        System.out.println("A:\n" + src + "\n");
        System.out.println("U:\n" + U + "\n");
        System.out.println("T:\n" + T + "\n");
        System.out.println("UTU^H:\n" + U.mult(T).mult(U.H()) + "\n");

//        CMatrix[] TU = SchurDecomposition.real2ComplexSchur(T.toReal(), U.toReal());
//
//        System.out.println("T complex:\n" + TU[0] + "\n");
//        System.out.println("U complex:\n" + TU[1] + "\n");
//        System.out.println("complex UTU^H:\n" + TU[1].mult(TU[0]).mult(TU[1].H()) + "\n");

//        System.out.println(Math.ulp(1.0));
    }
}
