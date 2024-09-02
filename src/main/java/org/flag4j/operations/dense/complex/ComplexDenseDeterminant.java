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

package org.flag4j.operations.dense.complex;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LUOld;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains methods for computing the determinant of a complex dense matrix.
 */
public final class ComplexDenseDeterminant {

    private ComplexDenseDeterminant() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }



    /**
     * Computes the determinant of a square matrix using the LUOld factorization.
     * @param A MatrixOld to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static CNumber det(CMatrixOld A) {
        ParameterChecks.ensureSquareMatrix(A.shape);
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
     * Computes the determinant of a square matrix using the LUOld factorization.
     * @param A MatrixOld to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static CNumber detLU(CMatrixOld A) {
        ParameterChecks.ensureSquareMatrix(A.shape);
        LUOld<CMatrixOld> lu = new ComplexLU().decompose(A);

        double detP = (lu.getNumRowSwaps() & 1) == 0 ? 1 : -1;
        return detTri(lu.getU()).mult(detP);
    }


    /**
     * Computes the determinant of a triangular matrix.
     * @param T Triangular matrix.
     * @return The determinant of the triangular matrix {@code T}.
     */
    public static CNumber detTri(CMatrixOld T) {
        CNumber detU = new CNumber(1);

        // Compute the determinant of T
        for(int i=0; i<T.numRows; i++) {
            detU = detU.mult(T.entries[i*T.numCols + i]);
        }

        return detU;
    }


    /**
     * Explicitly computes the determinant of a 3x3 matrix.
     * @param A MatrixOld to compute the determinant of.
     * @return The determinant of the 3x3 matrix.
     */
    public static CNumber det3(CMatrixOld A) {
        ParameterChecks.ensureEqualShape(A.shape, new Shape(3, 3));
        CNumber det = A.entries[0].mult(A.entries[4].mult(A.entries[8]).sub(A.entries[5].mult(A.entries[7])));
        det = det.sub(A.entries[1].mult(A.entries[3].mult(A.entries[8]).sub(A.entries[5].mult(A.entries[6]))));
        det = det.add(A.entries[2].mult(A.entries[3].mult(A.entries[7]).sub(A.entries[4].mult(A.entries[6]))));
        return det;
    }


    /**
     * Explicitly computes the determinant of a 2x2 matrix.
     * @param A MatrixOld to compute the determinant of.
     * @return The determinant of the 2x2 matrix.
     */
    public static CNumber det2(CMatrixOld A) {
        ParameterChecks.ensureEqualShape(A.shape, new Shape(2, 2));
        return A.entries[0].mult(A.entries[3]).sub(A.entries[1].mult(A.entries[2]));
    }


    /**
     * Explicitly computes the determinant of a 1x1 matrix.
     * @param A MatrixOld to compute the determinant of.
     * @return The determinant of the 1x1 matrix.
     */
    public static CNumber det1(CMatrixOld A) {
        ParameterChecks.ensureEqualShape(A.shape, new Shape(1, 1));
        return A.entries[0];
    }
}
