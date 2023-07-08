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
        super(true);
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

        T = workT; // Update the T matrix.

        if(computeU) {
            // Collect Hessenburg decomposition similarity transformations.
            U = hess.Q.mult(workU);
        }

        return this;
    }


    /**
     * Computes the Wilkinson shift for a complex matrix. That is, the eigenvalue of the lower left 2x2 sub matrix
     * which is closest in magnitude to the lower left entry of the matrix.
     * @return The Wilkinson shift for the {@code src} matrix.
     */
    private CNumber getWilkinsonShift() {
        // Compute eigenvalues of lower 2x2 block.
        CVector lambdas = Eigen.get2x2LowerRightBlockEigenValues(workT);
        CNumber v = workT.entries[workT.entries.length-1];

        if(v.sub(lambdas.entries[0]).mag() < v.sub(lambdas.entries[1]).mag()) {
            return lambdas.entries[0];
        } else {
            return lambdas.entries[1];
        }
    }


    /**
     * Applies an implicit double shift QR iteration using the generalized Rayleigh quotient shift
     */
    @Override
    protected void applyDoubleShift() {
        // Compute eigenvalues of lower right 2x2 block.
        CVector rho = Eigen.get2x2LowerRightBlockEigenValues(workT);

        CNumber a = workT.entries[0];
        CNumber b = workT.entries[1];
        CNumber c = workT.entries[workT.numCols];
        CNumber d = workT.entries[workT.numCols + 1];
        CNumber e = workT.entries[2*workT.numCols + 1];

        CVector p = new CVector(
                a.sub(rho.get(0)).mult( a.sub(rho.get(1)) ).add( b.mult(c) ),
                c.mult( a.add(d).sub(rho.get(0)).sub(rho.get(1)) ),
                e.mult(c)
        );

        // Apply reflector and introduce bulge.
        applyTransforms(Householder.getReflector(p));
        chaseBulge(); // Chase the bulge.
    }


    /**
     * Applies a single shift implicit QR iteration using the Wilkinson shift.
     */
    @Override
    protected void applySingleShift() {
        applySingleShift(getWilkinsonShift());
    }


    /**
     * Applies a single shift implicit QR iteration with a specified shift.
     * @param shift Shift to use for the single shift implicit QR iteration.
     */
    private void applySingleShift(CNumber shift) {
        CVector p = new CVector(
                workT.entries[0].sub(shift),
                workT.entries[workT.numCols]
        );

        CMatrix ref = Householder.getReflector(p);

        applyTransforms(ref); // Apply reflector to T and introduce bulge.
        chaseBulge(); // Chase the bulge from T.
    }


    /**
     * Applies a single shift implicit QR iterations with a random shift.
     * @param m Lower right index of the sub-matrix currently being worked on.
     */
    @Override
    protected void applyExceptionalShift(int m) {
        applySingleShift(getExceptionalShift(m));
    }


    /**
     * Initializes the unitary matrix in the schur decomposition.
     *
     * @return An initial {@code U} matrix. i.e. an identity matrix.
     */
    @Override
    protected CMatrix initU() {
        return CMatrix.I(workT.numRows);
    }


    /**
     * Deflates the {@code T} matrix in the decomposition and updates {@code U} if needed.
     *
     * @param m Row and column index of the lower right entry of the 2x2 block to deflate in {@code T}.
     */
    @Override
    protected void deflateT(int m) {
        deflateT(workT, workU, m);
    }


    /**
     * Chases the bulge from the {@code T} matrix introduced by the unitary transformations and returns {@code T}
     * to upper Hessenburg form.
     */
    protected void chaseBulge() {
        HessenburgDecomposition<CMatrix, CVector> hess = new ComplexHessenburgDecomposition(computeU);

        // TODO: Replace this with a proper bulge chaise that takes advantage of T being nearly upper Hessenburg already
        workT = hess.decompose(workT).getH(); // A crude bulge chase using the Hessenburg decomposition.

        if(computeU) {
            // Collect similarity transformations.
            workU = workU.mult(hess.getQ());
        }
    }


    /**
     * Constructs a shift in a random direction which is of the same magnitude as the elements
     * in the {@link #T} matrix.
     * @param m Lower right index of the sub-matrix currently being worked on.
     * @return A shift in a random direction which is of the same magnitude as the elements
     * in the {@link #T} matrix.
     */
    private CNumber getExceptionalShift(int m) {
        return rand.random(workT.get(m, m).mag());
    }


    /**
     * Checks if an entry along the first sub-diagonal of the {@code T} matrix in the Schur decomposition has
     * converged to zero within machine precision. A value is considered to be converged if it is small relative to the
     * two values on the diagonal of the {@code T} matrix immediately next to the value of interest. That is, a
     * value at index {@code (m, m-1)} is considered converged if it is less in absolute value than the absolute
     * sum of the entries at {@code (m-1, m-1)} and {@code (m, m)} times machine epsilon
     * (i.e. {@link Math#ulp(double)  Math.ulp(1.0d)}).
     * @param m Row index of the value of interest within the {@code T} matrix.
     * @return True if the specified entry has not converged. That is, the entry in the {@code T} matrix is greater
     * than (in absolute value) machine precision (i.e. Math.ulp(1.0)) times the absolute sum of the entries along the
     * block 2x2 matrix on the diagonal of {@code T} containing the entry. Otherwise, returns false.
     */
    @Override
    protected boolean notConverged(int m) {
        return notConverged(workT, m);
    }
}
