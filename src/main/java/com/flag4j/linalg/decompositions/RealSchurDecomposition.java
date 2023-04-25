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


/**
 * <p>This class computes the Schur decomposition of a square matrix.
 * That is, decompose a square matrix {@code A} into {@code A=QUQ<sup>H</sup>} where {@code Q} is a unitary
 * matrix whose columns are the eigenvectors of {@code A} and {@code U} is an upper triangular matrix in
 * Schur form whose diagonal entries are the eigenvalues of {@code A}, corresponding to the columns of {@code Q},
 * repeated per their multiplicity.</p>
 *
 * <p>Note, even if a matrix has only real entries, both {@code Q} and {@code U} may contain complex values.</p>
 */
public class RealSchurDecomposition extends SchurDecomposition<Matrix> {

    /**
     * Creates a Schur decomposer for a real dense matrix.
     */
    public RealSchurDecomposition() {
        super();
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(computeU);
    }


    /**
     * Creates a decomposer for a real dense matrix to compute the Schur decomposition which
     * will run for at most {@code maxIterations} iterations.
     * @param maxIterations Maximum number of iterations to run the QR algorithm for when computing the
     *                      Schur decomposition.
     */
    protected RealSchurDecomposition(int maxIterations) {
        super(maxIterations);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(computeU);
    }


    /**
     * Creates a decomposer for a real dense matrix to compute the Schur decomposition which will
     * run for at most {@code maxIterations} iterations.
     * @param maxIterations Maximum number of iterations to run the QR algorithm for when computing the
     *                      Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    protected RealSchurDecomposition(boolean computeU, int maxIterations) {
        super(computeU, maxIterations);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(computeU);
    }


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeU A flag which indicates if the unitary matrix {@code Q} should be computed.<br>
     *                 - If true, the {@code Q} matrix will be computed.
     *                 - If false, the {@code Q} matrix will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    protected RealSchurDecomposition(boolean computeU) {
        super(computeU);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(computeU);
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
        // TODO: Add balancing before converting to Hessenburg. See Fundamentals of Matrix Computations Watkins, p342.
        hess.decompose(src); // Compute a Hessenburg matrix which is similar to src (i.e. has the same eigenvalues).

        System.out.println("hessH:\n" + hess.getH() + "\n");
        System.out.println("hessQ:\n" + hess.getQ() + "\n");
        System.out.println("hess QHQ^T:\n" + hess.getQ().mult(hess.getH()).multTranspose(hess.getQ()) + "\n");

        shiftedQR(hess.getH().toComplex()); // Compute Schur decomposition of the Hessenburg form.
//        realShiftedQR(hess.getH()); // Compute real Schur decomposition of the Hessenburg form.

        if(computeU) {
            U = hess.getQ().mult(U); // Convert Hessenburg eigenvectors to the eigenvectors of the source matrix.
        }

        return this;
    }


    public RealSchurDecomposition doubleShiftImplicitQR(Matrix src) {
        debug = false;
        hess.decompose(src); // Compute a Hessenburg matrix which is similar to src (i.e. has the same eigenvalues).
        doubleShiftImplicitQR(hess.getH().toComplex());

        if(computeU) {
            U = hess.Q.mult(U); // Convert Hessenburg eigenvectors to the eigenvectors of the source matrix.
        }

        return this;
    }


    /**
     * Computes the Schur decomposition, in real Schur form, for a real dense matrix using the shifted QR algorithm
     * (Rayleigh shift).
     * @param H The matrix to compute the Schur decomposition of. Assumed to be in upper Hessenburg form.
     */
    private void realShiftedQR(Matrix H) {
        if(useDefaultMaxIterations) {
            // The algorithm should converge within machine precision in O(n^3).
            maxIterations = (int) Math.max(Math.pow(H.numRows, 3), MIN_DEFAULT_ITERATIONS);
        }

        int count = 0;
        int n = H.numRows-1;

        Matrix Treal = H;
        Matrix Ureal = Matrix.I(H.numRows);

        RealQRDecomposition qr = new RealQRDecomposition(); // Decomposer for use in the QR algorithm.
        Matrix Q; // Q matrix from QR decomposition.
        Matrix R; // R matrix from QR decomposition.

        Matrix mu;

        // Apply the QR algorithm (Shifted QR algorithm using Rayleigh shift).
        while(count<maxIterations) {
            count++;
            mu = Matrix.I(n+1).mult(Treal.entries[Treal.entries.length-1]); // Construct diagonal matrix.

            qr.decompose(Treal.sub(mu)); // Compute the QR decomposition with a shift.
            Q = qr.getQ();
            R = qr.getR();

            Treal = R.mult(Q).add(mu); // Reverse the shift.
            Ureal = Ureal.mult(Q);
        }

        U = Ureal.toComplex();
        T = Treal.toComplex();
    }


    public static void main(String[] args) {
        double[][] aEntries = {
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        Matrix A = new Matrix(aEntries);

        double[][] bEntries = {
                {1, 2, 5},
                {2, 3, 4},
                {5, 4, 9}};
        Matrix B = new Matrix(bEntries);

        double[][] cEntries = {
                {1, 2, 1},
                {2, 3, 4},
                {5, 12.1, 9}};
        Matrix C = new Matrix(cEntries);

        double[][] dEntries = {
                {1, -1},
                {1, 1}};
        Matrix D = new Matrix(dEntries);

        double[][] eEntries = {
                {0, 0, -1},
                {1, 0, 0},
                {0, 1, 0}};
        Matrix E = new Matrix(eEntries);

        double[][] fEntries = {
                {2, 1, 0, 0, 0, 0},
                {1, 2, 1, 0, 0, 0},
                {0, 1, 2, 1, 0, 0},
                {0, 0, 1, 2, 1, 0},
                {0, 0, 0, 1, 2, 1},
                {0, 0, 0, 0, 1, 2}};
        Matrix F = new Matrix(fEntries);

        Matrix src = F;

        RealSchurDecomposition schur = new RealSchurDecomposition();
        schur.doubleShiftImplicitQR(src);

        CMatrix T = schur.getT();
        CMatrix U = schur.getU();

        System.out.println("A:\n" + src + "\n");
        System.out.println("T:\n" + T + "\n");
        System.out.println("U:\n" + U + "\n");
        System.out.println("UTU^H:\n" + U.mult(T).mult(U.H()));
    }
}
