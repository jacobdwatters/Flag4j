/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.RealLUDecomposition;
import com.flag4j.linalg.decompositions.RealQRDecomposition;
import com.flag4j.util.ErrorMessages;


/**
 * This class provides methods for several matrix decompositions.
 */
public class Decompose {

    /**
     * Object for computing LU decomposition.
     */
    private static final RealLUDecomposition LU = new RealLUDecomposition();

    /**
     * Object for computing full QR decomposition.
     */
    private static final RealQRDecomposition QR = new RealQRDecomposition();


    private Decompose() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * <p>Computes the {@link RealLUDecomposition LU factorization} of a matrix using partial pivoting. That is, decomposes an m-by-n matrix {@code A} into {@code PA=LU}
     * where {@code P} is a permutation matrix, {@code L} is a unit lower triangular matrix, and U is an upper triangular matrix.</p>
     * @param A Matrix to decompose.
     * @return Returns an array of matrices containing in order {@code {P, L, U}} corresponding to {@code PA=LU}.
     */
    public static Matrix[] LU(Matrix A) {
        LU.decompose(A);
        return new Matrix[]{LU.getP(), LU.getL(), LU.getU()};
    }


    /**
     * Compute the full {@link RealQRDecomposition QR factorization} of a matrix using Householder reflectors. That is, decomposes an m-by-n matrix
     * {@code A} into {@code A=QR} where {@code Q} is an orthogonal matrix and {@code R} is an {@link Matrix#isTriU() upper triangular} matrix.
     * @param A Matrix to decompose.
     * @return Returns an array of matrices containing in order {@code {Q, R}} corresponding to {@code A=QR}.
     */
    public static Matrix[] QR(Matrix A) {
        QR.decompose(A);
        return new Matrix[]{QR.getQ(), QR.getR()};
    }
}
