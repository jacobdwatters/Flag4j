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

package com.flag4j.operations.dense.complex;


import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.Shape;
import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;
import com.flag4j.linalg.Decompose;
import com.flag4j.linalg.decompositions.lu.LUDecomposition;
import com.flag4j.linalg.decompositions.lu.RealLUDecomposition;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains methods for computing the determinant of a complex dense matrix.
 */
public class ComplexDenseDeterminant {

    private ComplexDenseDeterminant() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }



    /**
     * Computes the determinant of a square matrix using the LU factorization.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static CNumber det(CMatrix A) {
        ParameterChecks.assertSquareMatrix(A.shape);
        CNumber det;

        switch (A.numRows) {
            case 1:
                det = det1(A);
                break;
            case 2:
                det = det2(A);
                break;
            case 3:
                det = det3(A);
                break;
            default:
                det = detLU(A);
                break;
        }

        return det;
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static CNumber detLU(CMatrix A) {
        ParameterChecks.assertSquareMatrix(A.shape);
        CMatrix[] LU = Decompose.lu(A);

        return detLU(LU[0].toReal(), LU[1], LU[2]);
    }


    /**
     * Computes the determinant for a matrix which has been factored into a unit lower triangular matrix {@code L}
     * and an upper triangular matrix {@code U} using partial pivoting (i.e. row swaps).
     * @param P Row permutation matrix in the {@code LU} decomposition.
     * @param L Unit lower triangular matrix.
     * @param U Upper triangular matrix.
     * @return The determinant of the matrix which has been factored into a unit lower triangular matrix {@code L}
     * and an upper triangular matrix {@code U} using partial pivoting.
     */
    public static CNumber detLU(Matrix P, CMatrix L, CMatrix U) {
        int numSwaps = (int) (L.numRows - P.trace()) - 1; // Number of swaps in permutation matrix.
        double detP = Math.pow(-1, numSwaps); // Compute the determinant of P.
        return detLU(L, U).mult(detP);
    }


    /**
     * Computes the determinant for a matrix which has been factored into a unit lower triangular matrix {@code L}
     * and an upper triangular matrix {@code U} with no pivoting.
     * @param L Unit lower triangular matrix.
     * @param U Upper triangular matrix.
     * @return The determinant of the matrix which has been factored into a unit lower triangular matrix {@code L}
     * and an upper triangular matrix {@code U}.
     * @see LUDecomposition
     * @see RealLUDecomposition
     */
    public static CNumber detLU(CMatrix L, CMatrix U) {
        CNumber detU = new CNumber(1);

        // Compute the determinant of U
        for(int i=0; i<U.numRows; i++) {
            detU.multEq(U.entries[i*U.numCols + i]);
        }

        return detU;
    }


    /**
     * Explicitly computes the determinant of a 3x3 matrix.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 3x3 matrix.
     */
    public static CNumber det3(CMatrix A) {
        ParameterChecks.assertEqualShape(A.shape, new Shape(3, 3));
        CNumber det = A.entries[0].mult(A.entries[4].mult(A.entries[8]).sub(A.entries[5].mult(A.entries[7])));
        det.subEq(A.entries[1].mult(A.entries[3].mult(A.entries[8]).sub(A.entries[5].mult(A.entries[6]))));
        det.addEq(A.entries[2].mult(A.entries[3].mult(A.entries[7]).sub(A.entries[4].mult(A.entries[6]))));
        return det;
    }


    /**
     * Explicitly computes the determinant of a 2x2 matrix.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 2x2 matrix.
     */
    public static CNumber det2(CMatrix A) {
        ParameterChecks.assertEqualShape(A.shape, new Shape(2, 2));
        return A.entries[0].mult(A.entries[3]).sub(A.entries[1].mult(A.entries[2]));
    }


    /**
     * Explicitly computes the determinant of a 1x1 matrix.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 1x1 matrix.
     */
    public static CNumber det1(CMatrix A) {
        ParameterChecks.assertEqualShape(A.shape, new Shape(1, 1));
        return A.entries[0].copy();
    }
}
