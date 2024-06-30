/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.flag4j.linalg.decompositions.chol.Cholesky;
import org.flag4j.linalg.decompositions.chol.RealCholesky;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.decompositions.lu.RealLU;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.decompositions.svd.SVD;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.ComplexForwardSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealForwardSolver;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * This class provides methods for computing the inverse of a matrix. Specialized methods are provided for inverting triangular,
 * diagonal, and symmetric positive definite matrices.
 */
public class Invert {

    private Invert() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the inverse of this matrix. This is done by computing the {@link LU LU decomposition} of
     * this matrix, inverting {@code L} using a back-solve algorithm, then solving {@code U*inv(src)=inv(L)}
     * for {@code inv(src)}.
     *
     * @param src Matrix to compute inverse of.
     * @return The inverse of this matrix.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular (i.e. not invertible).
     */
    public static Matrix inv(Matrix src) {
        ParameterChecks.assertSquareMatrix(src.shape);
        LU<Matrix> lu = new RealLU().decompose(src);

        // Solve U*inv(A) = inv(L) for inv(A)
        RealBackSolver backSolver = new RealBackSolver();
        RealForwardSolver forwardSolver = new RealForwardSolver(true);

        // Compute the inverse of unit lower triangular matrix L.
        Matrix Linv = forwardSolver.solveIdentity(lu.getL());
        Matrix inverse = backSolver.solveLower(lu.getU(), Linv); // Compute inverse of row permuted A.

        return lu.getP().rightMult(inverse); // Finally, apply permutation matrix from LU decomposition.
    }


    /**
     * Computes the inverse of this matrix. This is done by computing the {@link LU LU decomposition} of
     * this matrix, inverting {@code L} using a back-solve algorithm, then solving {@code U*inv(src)=inv(L)}
     * for {@code inv(src)}.
     *
     * @param src Matrix to compute inverse of.
     * @return The inverse of this matrix.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular (i.e. not invertible).
     */
    public static CMatrix inv(CMatrix src) {
        ParameterChecks.assertSquareMatrix(src.shape);
        LU<CMatrix> lu = new ComplexLU().decompose(src);

        // Solve U*inv(A) = inv(L) for inv(A)
        ComplexBackSolver backSolver = new ComplexBackSolver();
        ComplexForwardSolver forwardSolver = new ComplexForwardSolver(true);

        // Compute the inverse of unit lower triangular matrix L.
        CMatrix Linv = forwardSolver.solveIdentity(lu.getL());
        CMatrix inverse = backSolver.solveLower(lu.getU(), Linv); // Compute inverse of row permuted A.

        return lu.getP().rightMult(inverse); // Finally, apply permutation matrix from LU decomposition.
    }


    /**
     * Inverts an upper triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * upper triangular.
     * @param src Upper triangular matrix to compute the inverse of.
     * @return The inverse of the upper triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static Matrix invTriU(Matrix src) {
        return new RealBackSolver().solveIdentity(src); // If the matrix is singular, it will be caught here.
    }


    /**
     * Inverts a lower triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * lower triangular.
     * @param src Lower triangular matrix to compute the inverse of.
     * @return The inverse of the lower triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static Matrix invTriL(Matrix src) {
        return new RealForwardSolver().solveIdentity(src); // If the matrix is singular, it will be caught here.
    }


    /**
     * Inverts a diagonal matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * diagonal.
     * @param src Diagonal matrix to compute the inverse of.
     * @return The inverse of the diagonal matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static Matrix invDiag(Matrix src) {
        ParameterChecks.assertSquareMatrix(src.shape);

        Matrix inverse = new Matrix(src.shape);

        double value;
        int step = src.numCols+1;
        double rank_condition = Math.ulp(1d);
        double det = 1;

        for(int i=0; i<src.numRows; i+=step) {
            value = src.entries[i];
            det *= value;
            inverse.entries[i] = 1.0/value;
        }

        if(Math.abs(det) <= rank_condition*Math.max(src.numRows, src.numCols)) {
            throw new SingularMatrixException("Could not invert.");
        }

        return inverse;
    }


    /**
     * Inverts an upper triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * upper triangular.
     * @param src Upper triangular matrix to compute the inverse of.
     * @return The inverse of the upper triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static CMatrix invTriU(CMatrix src) {
        return new ComplexBackSolver().solveIdentity(src); // If matrix is singular, it will be caught here.
    }


    /**
     * Inverts a lower triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * lower triangular.
     * @param src Lower triangular matrix to compute the inverse of.
     * @return The inverse of the lower triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static CMatrix invTriL(CMatrix src) {
        return new ComplexForwardSolver().solveIdentity(src); // If matrix is singular, it will be caught here.
    }


    /**
     * Inverts a diagonal matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * diagonal.
     * @param src Diagonal matrix to compute the inverse of.
     * @return The inverse of the diagonal matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    public static CMatrix invDiag(CMatrix src) {
        ParameterChecks.assertSquareMatrix(src.shape);

        CMatrix inverse = new CMatrix(src.shape);

        CNumber value;
        int step = src.numCols+1;
        double rank_condition = Math.ulp(1.0d);
        CNumber det = CNumber.one();

        for(int i=0; i<src.numRows; i+=step) {
            value = src.entries[i];
            det.multEq(value);
            inverse.entries[i] = value.multInv();
        }

        if(det.mag() <= rank_condition*Math.max(src.numRows, src.numCols)) {
            throw new SingularMatrixException("Could not invert.");
        }

        return inverse;
    }


    /**
     * Inverts a symmetric positive definite matrix.
     * @param src Positive definite matrix. It will not be verified if {@code src} is actually symmetric positive definite.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     * @see #invSymPosDef(Matrix, boolean)
     */
    public static Matrix invSymPosDef(Matrix src) {
        return invSymPosDef(src, false);
    }


    /**
     * Inverts a symmetric positive definite matrix.
     * @param src Positive definite matrix.
     * @param checkPosDef Flag indicating if a check should be made to see if {@code src} is actually symmetric
     *                    positive definite. <b>WARNING</b>: Checking if the matrix is positive definite can be very computationally
     *                    expensive.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     */
    public static Matrix invSymPosDef(Matrix src, boolean checkPosDef) {
        Cholesky<Matrix> chol = new RealCholesky(checkPosDef).decompose(src);
        RealBackSolver backSolver = new RealBackSolver();
        RealForwardSolver forwardSolver = new RealForwardSolver(true);

        // Compute the inverse of unit lower triangular matrix L.
        Matrix Linv = forwardSolver.solveIdentity(chol.getL());

        return backSolver.solveLower(chol.getLH(), Linv); // Compute inverse of src.
    }

    // ------------------------------------------- Pseudo-inverses below -------------------------------------------

    /**
     * Computes the pseudo-inverse of this matrix. That is, for a matrix {@code A}, computes the Moore–Penrose
     * {@code A}<sup>+</sup> such that the following hold:
     * <ol>
     *   <li>{@code AA}<sup>+</sup>{@code A=A}.</li>
     *   <li>{@code A}<sup>+</sup>{@code AA}<sup>+</sup>{@code =A}<sup>+</sup>.</li>
     *   <li>{@code AA}<sup>+</sup> is Hermation.</li>
     *   <li>{@code A}<sup>+</sup>{@code A} is also Hermation.</li>
     * </ol>
     *
     * @return The Moore–Penrose pseudo-inverse of this matrix.
     */
    public static Matrix pInv(Matrix src) {
        SVD<Matrix> svd = new RealSVD().decompose(src);
        Matrix sInv = Invert.invDiag(svd.getS());

        return svd.getV().mult(sInv).mult(svd.getU().T());
    }


    /**
     * Computes the pseudo-inverse of this matrix. That is, for a matrix {@code A}, computes the Moore–Penrose
     * {@code A}<sup>+</sup> such that the following hold:
     * <ol>
     *   <li>{@code AA}<sup>+</sup>{@code A=A}.</li>
     *   <li>{@code A}<sup>+</sup>{@code AA}<sup>+</sup>{@code =A}<sup>+</sup>.</li>
     *   <li>{@code AA}<sup>+</sup> is Hermation.</li>
     *   <li>{@code A}<sup>+</sup>{@code A} is also Hermation.</li>
     * </ol>
     *
     * @return The Moore–Penrose pseudo-inverse of this matrix.
     */
    public CMatrix pInv(CMatrix src) {
        SVD<CMatrix> svd = new ComplexSVD().decompose(src);
        Matrix sInv = Invert.invDiag(svd.getS());

        return svd.getV().mult(sInv).mult(svd.getU().H());
    }
}