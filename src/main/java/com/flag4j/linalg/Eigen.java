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
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.decompositions.RealSchurDecomposition;
import com.flag4j.linalg.decompositions.SchurDecomposition;

/**
 * This class provides several methods useful for computing eigen values, eigen vectors, as well as singular values and
 * singular vectors.
 */
public class Eigen {

    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(Matrix src) {
        CVector lambda = get2x2EigenValues(src.toComplex());
        return lambda;
    }


    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(CMatrix src) {
        // TODO: While theoretically correct, there are some numerical issues here.
        CVector lambda = new CVector(2);
        int n = src.numRows-1;

        // Get the four entries from lower right 2x2 sub-matrix.
        CNumber a = src.entries[0];
        CNumber b = src.entries[1];
        CNumber c = src.entries[2];
        CNumber d = src.entries[3];

        CNumber det = a.mult(d).sub(b.mult(c)); // 2x2 determinant.
        CNumber htr = a.add(b).div(2); // Half of the 2x2 trace.

        // 2x2 block eigenvalues.
        lambda.entries[0] = htr.add(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));
        lambda.entries[1] = htr.sub(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));

        return lambda;
    }


    /**
     * Computes the eigenvalues for the lower right 2x2 block matrix of a larger matrix.
     * @param src Source matrix to compute eigenvalues of lower right 2x2 block.
     * @return A vector of length 2 containing the eigenvalues of the lower right 2x2 block of {@code src}.
     */
    public static CVector get2x2LowerLeftBlockEigenValues(CMatrix src) {
        // TODO: While theoretically correct, there are some numerical
        //  issues here.
        CVector shifts = new CVector(2);
        int n = src.numRows-1;

        // Get the four entries from lower right 2x2 sub-matrix.
        CNumber a = src.entries[(n-1)*(src.numCols + 1)];
        CNumber b = src.entries[(n-1)*src.numCols + n];
        CNumber c = src.entries[n*(src.numCols + 1) - 1];
        CNumber d = src.entries[(n)*(src.numCols + 1)];

        CNumber det = a.mult(d).sub(b.mult(c)); // 2x2 determinant.
        CNumber htr = a.add(b).div(2); // Half of the 2x2 trace.

        // 2x2 block eigenvalues.
        shifts.entries[0] = htr.add(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));
        shifts.entries[1] = htr.sub(CNumber.sqrt(CNumber.pow(htr, 2).sub(det)));

        return shifts;
    }


    /**
     * Computes the eigenvalues of a square real dense matrix.
     * @param src The matrix to compute the eigenvalues of.
     * @return The eigenvalues of the {@code src} matrix.
     */
    public static CVector getEigenValues(Matrix src) {
        CVector lambdas = new CVector(src.numRows);

        SchurDecomposition<Matrix> schur = new RealSchurDecomposition(false).decompose(src);
        CMatrix T = schur.getT();

        // Extract diagonal of T.
        for(int i=0; i<T.numRows; i++) {
            lambdas.entries[i] = T.entries[i*(T.numCols + 1)];
        }

        return lambdas;
    }


    /**
     * Computes the eigenvectors of a square real dense matrix.
     * @param src The matrix to compute the eigenvectors of.
     * @return A matrix containing the eigenvectors of {@code src} as its columns.
     */
    public static CMatrix getEigenVectors(Matrix src) {
        CVector lambdas = new CVector(src.numRows);
        return new RealSchurDecomposition(true).decompose(src).getU();
    }
}
