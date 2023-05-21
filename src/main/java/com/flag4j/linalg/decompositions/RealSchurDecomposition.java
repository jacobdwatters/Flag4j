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
import com.flag4j.linalg.Eigen;
import com.flag4j.util.ParameterChecks;


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
        hess = new RealHessenburgDecomposition(super.computeU);
    }


    /**
     * Creates a decomposer for a real dense matrix to compute the Schur decomposition which
     * will run for at most {@code maxIterations} iterations.
     * @param maxIterations Maximum number of iterations to run the QR algorithm for when computing the
     *                      Schur decomposition.
     */
    public RealSchurDecomposition(int maxIterations) {
        super(maxIterations);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecomposition(super.computeU);
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
    public RealSchurDecomposition(boolean computeU, int maxIterations) {
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
    public RealSchurDecomposition(boolean computeU) {
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
        ParameterChecks.assertSquare(src.shape);
        hess.decompose(src); // Compute a Hessenburg matrix which is similar to src (i.e. has the same eigenvalues).g
        shiftedExplicitQR(hess.getH().toComplex());

        if(computeU) {
            U = hess.Q.mult(U); // Convert Hessenburg eigenvectors to the eigenvectors of the source matrix.
        }

        return this;
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
                {5.4, 4.0, 7.7},
                {3.5, -0.7, 2.8},
                {-3.2, 5.1, 0.8}};
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

        double[][] gEntries = {
                {1, 1, 1},
                {0, 2, 1},
                {0, 0, 3}};
        Matrix G = new Matrix(gEntries);

        Matrix src = C;

        CMatrix eigVectors = Eigen.getEigenVectors(src);

        System.out.println(eigVectors);
    }
}
