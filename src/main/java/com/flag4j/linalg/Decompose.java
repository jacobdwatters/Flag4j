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

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.cholesky.ComplexCholeskyDecomposition;
import com.flag4j.linalg.decompositions.cholesky.RealCholeskyDecomposition;
import com.flag4j.linalg.decompositions.lu.ComplexLUDecomposition;
import com.flag4j.linalg.decompositions.lu.RealLUDecomposition;
import com.flag4j.linalg.decompositions.qr.ComplexQRDecomposition;
import com.flag4j.linalg.decompositions.qr.RealQRDecomposition;
import com.flag4j.util.ErrorMessages;


/**
 * This class provides methods for several matrix decompositions.
 */
public final class Decompose {

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
    public static Matrix[] lu(Matrix A) {
        RealLUDecomposition LU = new RealLUDecomposition();
        LU.decompose(A);
        return new Matrix[]{LU.getP().toDense(), LU.getL(), LU.getU()};
    }


    /**
     * <p>Computes the {@link ComplexLUDecomposition LU factorization} of a complex matrix using partial pivoting.
     * That is, decomposes an m-by-n matrix {@code A} into {@code PA=LU} where {@code P} is a permutation matrix,
     * {@code L} is a unit lower triangular matrix, and U is an upper triangular matrix.</p>
     * @param A Matrix to decompose.
     * @return Returns an array of matrices containing in order {@code {P, L, U}} corresponding to {@code PA=LU}. Note,
     * although {@code P} is always real, it will be wrapped in a complex Matrix.
     */
    public static CMatrix[] lu(CMatrix A) {
        ComplexLUDecomposition LU = new ComplexLUDecomposition();
        LU.decompose(A);
        return new CMatrix[]{LU.getP().toDense().toComplex(), LU.getL(), LU.getU()};
    }


    /**
     * Compute the full {@link RealQRDecomposition QR factorization} of a matrix using Householder reflectors. That is, decomposes an m-by-n matrix
     * {@code A} into {@code A=QR} where {@code Q} is an {@link Matrix#isOrthogonal() orthogonal} matrix and {@code R} is an {@link Matrix#isTriU() upper triangular} matrix.
     * @param A Matrix to decompose.
     * @return Returns an array of matrices containing in order {@code {Q, R}} corresponding to {@code A=QR}.
     */
    public static Matrix[] qr(Matrix A) {
        RealQRDecomposition QR = new RealQRDecomposition();
        QR.decompose(A);
        return new Matrix[]{QR.getQ(), QR.getR()};
    }


    /**
     * Compute the full {@link ComplexQRDecomposition QR factorization} of a matrix using Householder reflectors. That is, decomposes an m-by-n matrix
     * {@code A} into {@code A=QR} where {@code Q} is {@link CMatrix#isUnitary() unitary} matrix and {@code R} is an {@link CMatrix#isTriU() upper triangular} matrix.
     * @param A Matrix to decompose.
     * @return Returns an array of matrices containing in order {@code {Q, R}} corresponding to {@code A=QR}.
     */
    public static CMatrix[] qr(CMatrix A) {
        ComplexQRDecomposition QR = new ComplexQRDecomposition();
        QR.decompose(A);
        return new CMatrix[]{QR.getQ(), QR.getR()};
    }


    /**
     * Computes the {@link RealCholeskyDecomposition cholescky decomposition} of a
     * symmetric positive-definite matrix. That is, decomposes a symmetric positive-definite matrix {@code A} into
     * {@code A=LL<sup>T</sup>} where {@code L} is a {@link Matrix#isTriL() lower triangular}.
     * @param src The matrix to decompose. Must be symmetric positive-definite.
     * @return The {@code L} matrix corresponding to {@code A=LL<sup>T</sup>}.
     * @throws IllegalArgumentException If {@code src} is not symmetric positive-definite.
     */
    public static Matrix cholesky(Matrix src) {
        RealCholeskyDecomposition cholesky = new RealCholeskyDecomposition();
        return cholesky.decompose(src).getL();
    }


    /**
     * Computes the {@link ComplexCholeskyDecomposition cholescky decomposition} of a
     * symmetric positive-definite matrix. That is, decomposes a symmetric positive-definite matrix {@code A} into
     * {@code A=LL<sup>*</sup>} where {@code L} is {@link CMatrix#isTriL() lower triangular}.
     * @param src The matrix to decompose. Must be symmetric positive-definite.
     * @return The {@code L} matrix corresponding to {@code A=LL<sup>*</sup>}.
     * @throws IllegalArgumentException If {@code src} is not symmetric positive-definite.
     */
    public static CMatrix cholesky(CMatrix src) {
        ComplexCholeskyDecomposition cholesky = new ComplexCholeskyDecomposition();
        return cholesky.decompose(src).getL();
    }
}
