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

package com.flag4j.linalg;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.linalg.decompositions.cholesky.CholeskyDecomposition;
import com.flag4j.linalg.decompositions.cholesky.RealCholeskyDecomposition;
import com.flag4j.linalg.solvers.ComplexBackSolver;
import com.flag4j.linalg.solvers.ComplexForwardSolver;
import com.flag4j.linalg.solvers.RealBackSolver;
import com.flag4j.linalg.solvers.RealForwardSolver;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class provides methods for computing the inverse of a matrix.
 */
public class Invert {

    private Invert() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
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
        ParameterChecks.assertSquare(src.shape);
        if(zeroOnDiag(src)) {
            throw new SingularMatrixException("Cannot invert.");
        }

        RealBackSolver backSolver = new RealBackSolver();
        return backSolver.solveIdentity(src);
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
        if(zeroOnDiag(src)) {
            throw new SingularMatrixException("Cannot invert.");
        }

        RealForwardSolver forwardSolver = new RealForwardSolver();
        return forwardSolver.solveIdentity(src);
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
        ParameterChecks.assertSquare(src.shape);

        Matrix inverse = new Matrix(src.shape);

        double value;
        int idx = 0;
        int step = src.numCols+1;

        for(int i=0; i<src.numRows; i++) {
            value = src.entries[idx];
            idx += step;

            if(value==0) {
                throw new SingularMatrixException("Cannot invert.");
            }

            inverse.entries[idx] = 1.0/value;
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
        ParameterChecks.assertSquare(src.shape);
        if(zeroOnDiag(src)) {
            throw new SingularMatrixException("Cannot invert.");
        }

        ComplexBackSolver backSolver = new ComplexBackSolver();
        return backSolver.solveIdentity(src);
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
        if(zeroOnDiag(src)) {
            throw new SingularMatrixException("Cannot invert.");
        }

        ComplexForwardSolver forwardSolver = new ComplexForwardSolver();
        return forwardSolver.solveIdentity(src);
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
        ParameterChecks.assertSquare(src.shape);

        CMatrix inverse = new CMatrix(src.shape);

        CNumber value;
        int idx = 0;
        int step = src.numCols+1;

        for(int i=0; i<src.numRows; i++) {
            value = src.entries[idx];
            idx += step;

            if(value.re==0 && value.im==0) {
                throw new SingularMatrixException("Cannot invert.");
            }

            inverse.entries[idx] = value.multInv();
        }

        return inverse;
    }


    /**
     * Checks if a matrix has a zero entry on the diagonal.
     * @param src Matrix of interest. Assumed to be square but not explicitly verified.
     * @return True if the {@code src} matrix has zeros on the diagonal. False otherwise.
     */
    private static boolean zeroOnDiag(Matrix src) {
        boolean result = false;

        double value;
        int idx = 0;
        int step = src.numCols+1;

        for(int i=0; i<src.numRows; i++) {
            value = src.entries[idx];
            idx += step;

            if(value==0) {
                result = true;
                break;
            }
        }

        return result;
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
        CholeskyDecomposition<Matrix> chol = new RealCholeskyDecomposition(checkPosDef).decompose(src);
        RealBackSolver backSolver = new RealBackSolver();
        RealForwardSolver forwardSolver = new RealForwardSolver(true);

        // Compute the inverse of unit lower triangular matrix L.
        Matrix Linv = forwardSolver.solveIdentity(chol.getL());

        return backSolver.solveLower(chol.getLH(), Linv); // Compute inverse of src.
    }


    /**
     * Checks if a matrix has a zero entry on the diagonal.
     * @param src Matrix of interest. Assumed to be square but not explicitly verified.
     * @return True if the {@code src} matrix has zeros on the diagonal. False otherwise.
     */
    private static boolean zeroOnDiag(CMatrix src) {
        boolean result = false;

        CNumber value;
        int idx = 0;
        int step = src.numCols+1;

        for(int i=0; i<src.numRows; i++) {
            value = src.entries[idx];
            idx += step;

            if(value.re==0 && value.im==0) {
                result = true;
                break;
            }
        }

        return result;
    }
}
