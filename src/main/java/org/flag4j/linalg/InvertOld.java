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

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.linalg.decompositions.chol.CholeskyOld;
import org.flag4j.linalg.decompositions.chol.ComplexCholeskyOld;
import org.flag4j.linalg.decompositions.chol.RealCholeskyOld;
import org.flag4j.linalg.decompositions.lu.ComplexLUOld;
import org.flag4j.linalg.decompositions.lu.LUOld;
import org.flag4j.linalg.decompositions.lu.RealLUOLd;
import org.flag4j.linalg.decompositions.svd.ComplexSVDOld;
import org.flag4j.linalg.decompositions.svd.RealSVDOld;
import org.flag4j.linalg.decompositions.svd.SVDOld;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolverOld;
import org.flag4j.linalg.solvers.exact.triangular.ComplexForwardSolverOld;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolverOld;
import org.flag4j.linalg.solvers.exact.triangular.RealForwardSolverOld;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.SingularMatrixException;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * This class provides methods for computing the inverse of a matrix. Specialized methods are provided for inverting triangular,
 * diagonal, and symmetric positive definite matrices.
 */
@Deprecated
public class InvertOld {

    private InvertOld() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the inverse of this matrix. This is done by computing the {@link LUOld LUOld decomposition} of
     * this matrix, inverting {@code L} using a back-solve algorithm, then solving {@code U*inv(src)=inv(L)}
     * for {@code inv(src)}.
     *
     * @param src MatrixOld to compute inverse of.
     * @return The inverse of this matrix.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular (i.e. not invertible).
     */
    @Deprecated
    public static MatrixOld inv(MatrixOld src) {
        ParameterChecks.ensureSquareMatrix(src.shape);
        LUOld<MatrixOld> lu = new RealLUOLd().decompose(src);

        // Solve U*inv(A) = inv(L) for inv(A)
        RealBackSolverOld backSolver = new RealBackSolverOld();
        RealForwardSolverOld forwardSolver = new RealForwardSolverOld(true);

        // Compute the inverse of unit lower triangular matrix L.
        MatrixOld Linv = forwardSolver.solveIdentity(lu.getL());
        MatrixOld inverse = backSolver.solveLower(lu.getU(), Linv); // Compute inverse of row permuted A.

        return lu.getP().rightMult(inverse); // Finally, apply permutation matrix from LUOld decomposition.
    }


    /**
     * Computes the inverse of this matrix. This is done by computing the {@link LUOld LUOld decomposition} of
     * this matrix, inverting {@code L} using a back-solve algorithm, then solving {@code U*inv(src)=inv(L)}
     * for {@code inv(src)}.
     *
     * @param src MatrixOld to compute inverse of.
     * @return The inverse of this matrix.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular (i.e. not invertible).
     */
    @Deprecated
    public static CMatrixOld inv(CMatrixOld src) {
        ParameterChecks.ensureSquareMatrix(src.shape);
        LUOld<CMatrixOld> lu = new ComplexLUOld().decompose(src);

        // Solve U*inv(A) = inv(L) for inv(A)
        ComplexBackSolverOld backSolver = new ComplexBackSolverOld();
        ComplexForwardSolverOld forwardSolver = new ComplexForwardSolverOld(true);

        // Compute the inverse of unit lower triangular matrix L.
        CMatrixOld Linv = forwardSolver.solveIdentity(lu.getL());
        CMatrixOld inverse = backSolver.solveLower(lu.getU(), Linv); // Compute inverse of row permuted A.

        return lu.getP().rightMult(inverse); // Finally, apply permutation matrix from LUOld decomposition.
    }


    /**
     * Inverts an upper triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * upper triangular.
     * @param src Upper triangular matrix to compute the inverse of.
     * @return The inverse of the upper triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    @Deprecated
    public static MatrixOld invTriU(MatrixOld src) {
        return new RealBackSolverOld().solveIdentity(src); // If the matrix is singular, it will be caught here.
    }


    /**
     * Inverts a lower triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * lower triangular.
     * @param src Lower triangular matrix to compute the inverse of.
     * @return The inverse of the lower triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    @Deprecated
    public static MatrixOld invTriL(MatrixOld src) {
        return new RealForwardSolverOld().solveIdentity(src); // If the matrix is singular, it will be caught here.
    }


    /**
     * Inverts a diagonal matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * diagonal.
     * @param src Diagonal matrix to compute the inverse of.
     * @return The inverse of the diagonal matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    @Deprecated
    public static MatrixOld invDiag(MatrixOld src) {
        ParameterChecks.ensureSquareMatrix(src.shape);
        MatrixOld inverse = new MatrixOld(src.shape);

        double value;
        int step = src.numCols+1;
        double det = 1;

        for(int i=0; i<src.entries.length; i+=step) {
            value = src.entries[i];
            det *= value;
            inverse.entries[i] = 1.0/value;
        }

        if(Math.abs(det) <= EPS_F64*Math.max(src.numRows, src.numCols)) {
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
    @Deprecated
    public static CMatrixOld invTriU(CMatrixOld src) {
        return new ComplexBackSolverOld().solveIdentity(src); // If matrix is singular, it will be caught here.
    }


    /**
     * Inverts a lower triangular matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * lower triangular and will treat it as such even if it is not triangular.
     * @param src Lower triangular matrix to compute the inverse of.
     * @return The inverse of the lower triangular matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    @Deprecated
    public static CMatrixOld invTriL(CMatrixOld src) {
        return new ComplexForwardSolverOld().solveIdentity(src); // If matrix is singular, it will be caught here.
    }


    /**
     * Inverts a diagonal matrix. <b>WARNING:</b> this method does not check that the matrix is actually
     * diagonal.
     * @param src Diagonal matrix to compute the inverse of.
     * @return The inverse of the diagonal matrix.
     * @throws SingularMatrixException If the matrix is singular (i.e. has at least one zero along the diagonal).
     * @throws IllegalArgumentException If the matrix is not square.
     */
    @Deprecated
    public static CMatrixOld invDiag(CMatrixOld src) {
        ParameterChecks.ensureSquareMatrix(src.shape);

        CMatrixOld inverse = new CMatrixOld(src.shape);

        CNumber value;
        int step = src.numCols+1;
        CNumber det = CNumber.ONE;

        for(int i=0; i<src.entries.length; i+=step) {
            value = src.entries[i];
            det = det.mult(value);
            inverse.entries[i] = value.multInv();
        }

        if(det.mag() <= EPS_F64*Math.max(src.numRows, src.numCols)) {
            throw new SingularMatrixException("Could not invert.");
        }

        return inverse;
    }


    /**
     * Inverts a symmetric positive definite matrix.
     * @param src Positive definite matrix. It will <i>not</i> be verified if {@code src} is actually symmetric positive definite.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     * @see #invSymPosDef(MatrixOld, boolean)
     */
    @Deprecated
    public static MatrixOld invSymPosDef(MatrixOld src) {
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
    @Deprecated
    public static MatrixOld invSymPosDef(MatrixOld src, boolean checkPosDef) {
        CholeskyOld<MatrixOld> chol = new RealCholeskyOld(checkPosDef).decompose(src);
        RealBackSolverOld backSolver = new RealBackSolverOld();
        RealForwardSolverOld forwardSolver = new RealForwardSolverOld();

        // Compute the inverse of lower triangular matrix L.
        MatrixOld Linv = forwardSolver.solveIdentity(chol.getL());

        return backSolver.solveLower(chol.getLH(), Linv); // Compute inverse of src.
    }


    /**
     * Inverts a hermitian positive definite matrix.
     * @param src Positive definite matrix. It will <i>not</i> be verified if {@code src} is actually hermitian positive definite.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     * @see #invSymPosDef(MatrixOld, boolean)
     */
    @Deprecated
    public static CMatrixOld invHermPosDef(CMatrixOld src) {
        return invHermPosDef(src, false);
    }


    /**
     * Inverts a hermitian positive definite matrix.
     * @param src Positive definite matrix.
     * @param checkPosDef Flag indicating if a check should be made to see if {@code src} is actually hermitian
     *                    positive definite. <b>WARNING</b>: Checking if the matrix is positive definite can be very computationally
     *                    expensive.
     * @return The inverse of the {@code src} matrix.
     * @throws IllegalArgumentException If the matrix is not square.
     * @throws SingularMatrixException If the {@code src} matrix is singular.
     */
    @Deprecated
    public static CMatrixOld invHermPosDef(CMatrixOld src, boolean checkPosDef) {
        CholeskyOld<CMatrixOld> chol = new ComplexCholeskyOld(checkPosDef).decompose(src);
        ComplexBackSolverOld backSolver = new ComplexBackSolverOld();
        ComplexForwardSolverOld forwardSolver = new ComplexForwardSolverOld();

        // Compute the inverse of lower triangular matrix L.
        CMatrixOld Linv = forwardSolver.solveIdentity(chol.getL());

        return backSolver.solveLower(chol.getLH(), Linv); // Compute inverse of src.
    }


    // ------------------------------------------- Pseudo-inverses below -------------------------------------------

    /**
     * Computes the pseudo-inverse of this matrix. That is, for a matrix {@code A}, computes the Moore–Penrose
     * {@code A}<sup>+</sup> such that the following hold:
     * <ol>
     *   <li>{@code AA}<sup>+</sup>{@code A=A}.</li>
     *   <li>{@code A}<sup>+</sup>{@code AA}<sup>+</sup>{@code =A}<sup>+</sup>.</li>
     *   <li>{@code AA}<sup>+</sup> is Hermitian.</li>
     *   <li>{@code A}<sup>+</sup>{@code A} is also Hermitian.</li>
     * </ol>
     *
     * @return The Moore–Penrose pseudo-inverse of this matrix.
     */
    @Deprecated
    public static MatrixOld pInv(MatrixOld src) {
        SVDOld<MatrixOld> svd = new RealSVDOld().decompose(src);
        MatrixOld sInv = InvertOld.invDiag(svd.getS());

        return svd.getV().mult(sInv).mult(svd.getU().T());
    }


    /**
     * Computes the pseudo-inverse of this matrix. That is, for a matrix {@code A}, computes the Moore–Penrose
     * {@code A}<sup>+</sup> such that the following hold:
     * <ol>
     *   <li>{@code AA}<sup>+</sup>{@code A=A}.</li>
     *   <li>{@code A}<sup>+</sup>{@code AA}<sup>+</sup>{@code =A}<sup>+</sup>.</li>
     *   <li>{@code AA}<sup>+</sup> is Hermitian.</li>
     *   <li>{@code A}<sup>+</sup>{@code A} is also Hermitian.</li>
     * </ol>
     *
     * @return The Moore–Penrose pseudo-inverse of this matrix.
     */
    @Deprecated
    public static CMatrixOld pInv(CMatrixOld src) {
        SVDOld<CMatrixOld> svd = new ComplexSVDOld().decompose(src);
        MatrixOld sInv = InvertOld.invDiag(svd.getS());

        return svd.getV().mult(sInv).mult(svd.getU().H());
    }

    // -------------------------------------------------------------------

    /**
     * Checks if matrices are inverses of each other. This method rounds values near zero to zero when checking
     * if the two matrices are inverses to account for floating point precision loss.
     *
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return True if matrix src2 is an inverse of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    @Deprecated
    public static boolean isInv(MatrixOld src1, MatrixOld src2) {
        boolean result;

        if(!src1.isSquare() || !src2.isSquare() || !src1.shape.equals(src2.shape)) {
            result = false;
        } else {
            result = src1.mult(src2).isCloseToI();
        }

        return result;
    }


    /**
     * Checks if matrices are inverses of each other.
     *
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return True if matrix src2 is an inverse (approximately) of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    @Deprecated
    public static boolean isInv(CMatrixOld src1, CMatrixOld src2) {
        boolean result;

        if(!src1.isSquare() || !src2.isSquare() || !src1.shape.equals(src2.shape)) {
            result = false;
        } else {
            result = src1.mult(src2).isCloseToI();
        }

        return result;
    }
}
