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

package com.flag4j.operations.dense.real;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.linalg.decompositions.LUDecomposition;
import com.flag4j.linalg.decompositions.RealLUDecomposition;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains methods for computing the determinant of a real dense matrix.
 */
public class RealDenseDeterminant {

    private RealDenseDeterminant() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static double det(Matrix A) {
        ParameterChecks.assertSquare(A.numRows, A.numCols);

        if (A.numRows == 1) {
            return A.entries[0];
        } else if (A.numRows == 2) {
            return A.entries[0] * A.entries[3] - A.entries[1] * A.entries[2];
        } else if (A.numRows == 3) {
            return det3(A);
        } else {
            LUDecomposition<Matrix> lu = new RealLUDecomposition().decompose(A);
            // Compute the determinant of P. (Check if lowest bit is zero to determine parity)
            double detP = (lu.getNumRowSwaps() & 1) == 0 ? 1 : -1;
            return detP * detTri(lu.getLU());
        }
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static double detLU(Matrix A) {
        ParameterChecks.assertSquare(A.numRows, A.numCols);

        RealLUDecomposition lu = new RealLUDecomposition();
        lu.decompose(A);
        double detP = (lu.getNumRowSwaps() & 1) == 0 ? 1 : -1;

        return detP * detTri(lu.getU());
    }


    /**
     * <p>
     * Computes the determinant for a matrix which has been factored into a unit lower triangular matrix {@code L}
     * and an upper triangular matrix {@code U} using partial pivoting (i.e. row swaps). Note, the matrix {@code L} is
     * not needed to compute this determinant since it is assumed to be unit lower triangular.
     * </p>
     *
     * <p>
     * Note, if the LU decomposition has not already been computed, simply use {@link #det(Matrix)}.
     * </p>
     *
     * @param P Row permutation matrix in the {@code LU} decomposition.
     * @param U Upper triangular matrix.
     * @return The determinant of the matrix which has been factored into a unit lower triangular matrix {@code L}
     * and an upper triangular matrix {@code U} using partial pivoting.
     */
    public static double detLU(Matrix P, Matrix U) {
        int numSwaps = (int) (P.numRows - P.trace()) - 1; // Number of swaps in permutation matrix.
        double detP = (numSwaps & 1) == 0 ? 1 : -1;
        return detP * detTri(U);
    }


    /**
     * Explicitly computes the determinant of a 3x3 matrix.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 3x3 matrix.
     */
    public static double det3(Matrix A) {
        ParameterChecks.assertEqualShape(A.shape, new Shape(3, 3));
        double det = A.entries[0] * (A.entries[4] * A.entries[8] - A.entries[5] * A.entries[7]);
        det -= A.entries[1] * (A.entries[3] * A.entries[8] - A.entries[5] * A.entries[6]);
        det += A.entries[2] * (A.entries[3] * A.entries[7] - A.entries[4] * A.entries[6]);
        return det;
    }


    /**
     * Explicitly computes the determinant of a 2x2 matrix.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 2x2 matrix.
     */
    public static double det2(Matrix A) {
        ParameterChecks.assertEqualShape(A.shape, new Shape(2, 2));
        return A.entries[0] * A.entries[3] - A.entries[1] * A.entries[2];
    }


    /**
     * Explicitly computes the determinant of a 1x1 matrix.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 1x1 matrix.
     */
    public static double det1(Matrix A) {
        ParameterChecks.assertEqualShape(A.shape, new Shape(1, 1));
        return A.entries[0];
    }


    /**
     * Computes the determinant for a triangular matrix. This method does not check that the matrix is actually
     * triangular.
     *
     * @param A Triangular matrix.
     * @return The determinant of the triangular matrix.
     */
    public static double detTri(Matrix A) {
        double det = 1;
        int step = A.numCols + 1;

        // Compute the determinant of U
        for (int i = 0; i < A.entries.length; i += step) {
            det *= A.entries[i];
        }

        return det;
    }
}
