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

package com.flag4j.linalg.decompositions.schur;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.Eigen;
import com.flag4j.linalg.decompositions.hess.RealHessenburgDecompositionOld;
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
     * Creates a Schur decomposer for a real dense matrix. {@code T} will be stored in
     * Real Schur form. To convert {@code T} to real schur from see
     * {@link SchurDecomposition#real2ComplexSchur(Matrix, Matrix)} or
     */
    public RealSchurDecomposition() {
        super(true);
        /* If there is no need to compute U in the Schur decomposition, there is no need to compute Q in the
           Hessenburg decomposition. */
        hess = new RealHessenburgDecompositionOld(super.computeU);
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
        hess = new RealHessenburgDecompositionOld(computeU);
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
        hess = new RealHessenburgDecompositionOld(computeU);
        this.realSchur = realSchur;
    }


    /**
     * Gets the real Schur form of the {@code T} matrix from the Schur decomposition.
     * @return The real Schur form of the {@code T} matrix from the Schur decomposition.
     */
    public Matrix getRealT() {
        return workT;
    }


    /**
     * Gets the {@code U} matrix in the Schur decomposition corresponding to the real
     * Schur form of the {@code T} matrix in the Schur decomposition.
     * @return The{@code U} matrix in the Schur decomposition corresponding to the real
     * Schur form of the {@code T} matrix in the Schur decomposition.
     */
    public Matrix getRealU() {
        return workU;
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

        // Collect Hessenburg decomposition similarity transformations.
        if(computeU) workU = hess.getQ().mult(workU);

        if(realSchur) {
            T = workT.toComplex();
            if(computeU) U = workU.toComplex();
        } else {
            // Convert to the complex schur form.
            CMatrix[] TU = SchurDecomposition.real2ComplexSchur(workT, workU);
            T = TU[0];
            if(computeU) U = TU[1];
        }

        return this;
    }


    /**
     * Applies an implicit double shift QR iteration using the generalized Rayleigh quotient shift.
     */
    @Override
    protected void applyDoubleShift() {
        // Compute eigenvalues of lower right 2x2 block.
        CNumber[] rho = Eigen.get2x2LowerRightBlockEigenValues(workT).entries;

        double p0, p1, p2;

        double a = workT.entries[0];
        double b = workT.entries[1];
        double c = workT.entries[workT.numCols];
        double d = workT.entries[workT.numCols + 1];

        // Check if the eigen values are complex conjugates.
        if(Math.abs(rho[0].im) > Math.ulp(1.0)) {
            double x = rho[0].re;
            double y = rho[0].im;

            p0 = a*(a - 2*x) + x*x + y*y + b*c;
            p1 = c*(a + d - 2*x);
        } else {
            p0 = (a - rho[0].re)*(a - rho[1].re) + b*c;
            p1 = c*(a + d - (rho[0].re + rho[1].re));
        }

        p2 = workT.entries[2*workT.numCols + 1]*c;

        // Apply reflector and introduce bulge.
        applyTransforms(Householder.getReflector(new Vector(p0, p1, p2)));
        chaseBulge(2); // Chase the bulge.
    }


    /**
     * Applies a single shift implicit QR iteration using the Rayleigh quotient shift.
     */
    @Override
    protected void applySingleShift() {
        // TODO: If the matrix is symmetric, the Wilkinson shift will be real and so that should be used instead.
        applySingleShift(workT.entries[workT.entries.length-1]);
    }


    /**
     * Applies a single shift implicit QR iteration with a specified shift.
     * @param shift Shift to use for the single shift implicit QR iteration.
     */
    private void applySingleShift(double shift) {
        Vector p = new Vector(
                workT.entries[0] - shift,
                workT.entries[workT.numCols]
        );
        Matrix ref = Householder.getReflector(p);

        applyTransforms(ref); // Apply reflector to workT and introduce bulge.
        chaseBulge(1); // Chase the bulge from workT.
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
    protected Matrix initU() {
        return Matrix.I(workT.numRows);
    }


    /**
     * Initializes a Householder reflector for use in the bulge chase.
     *
     * @param col Column vector to compute the Householder for.
     * @return A Householder reflector which zeros the values in {@code col} after
     * the first entry.
     */
    @Override
    protected Matrix initRef(Vector col) {
        return Householder.getReflector(col);
    }


    /**
     * Constructs a shift in a random direction which is of the same magnitude as the elements
     * in the {@link #workT} matrix.
     * @param m Lower right index of the sub-matrix currently being worked on.
     * @return A shift in a random direction which is of the same magnitude as the elements
     * in the {@link #workT} matrix.
     */
    private double getExceptionalShift(int m) {
        double shift = Math.abs(workT.entries[m*(workT.numCols + 1)]);
        shift = shift==0 ? 1 : shift; // Avoid using a zero shift.

        double p = 1.0 - Math.pow(0.1, numExceptionalShifts);
        shift *= p + 2.0*(1.0 - p)*(rand.nextDouble() - 0.5);

        shift = rand.nextBoolean() ? -shift : shift;

        return shift;
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
        return Math.abs(workT.entries[m*(workT.numCols + 1) - 1])
                > TOL*Math.abs( workT.entries[(m-1)*(workT.numCols + 1)]
                + Math.abs(workT.entries[m*(workT.numCols + 1)]) );
    }
}
