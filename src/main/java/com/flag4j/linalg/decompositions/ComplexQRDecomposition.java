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

package com.flag4j.linalg.decompositions;


import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.linalg.transformations.Householder;


/**
 * <p>Instances of this class compute the {@code QR} decomposition of a complex dense matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into an unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 */
public class ComplexQRDecomposition extends QRDecomposition<CMatrix> {

    /**
     * Constructs a {@code QR} decomposer which computes the full {@code QR} decomposition.
     */
    public ComplexQRDecomposition() {
        super();
    }


    /**
     * Constructs a {@code QR} decomposer which computes either the full or reduced {@code QR} decomposition.
     * @param fullQR Flag for determining if the full {@code QR} decomposition should be used.
     *               If true, the full {@code QR} decomposition will be computed, if false,
     *               the reduced {@code QR} decomposition will be computed.
     */
    public ComplexQRDecomposition(boolean fullQR) {
        super(fullQR);
    }


    /**
     * Computes the reduced QR decomposition on the src matrix.
     * @param src The source matrix to decompose.
     */
    @Override
    protected void reduced(CMatrix src) {
        full(src); // First compute the full decomposition

        int k = Math.min(src.numRows, src.numCols);

        // Now reduce the decomposition
        Q = Q.getSlice(0, src.numRows, 0, k);
        R = R.getSlice(0, k, 0, src.numCols);
    }


    /**
     * Computes the full QR decomposition on the src matrix.
     * @param src The source matrix to decompose.
     */
    @Override
    protected void full(CMatrix src) {
        R = new CMatrix(src); // Initialize R to the values in src.
        int m = R.numRows, n = R.numCols;
        int stop = Math.min(n, m-1);

        double eps = 1.0e-12;

        CMatrix H;
        CVector col;

        // Initialize Q to the identity matrix.
        Q = CMatrix.I(R.numRows);

        for(int i=0; i<stop; i++) {
            col = R.getColBelow(i, i).toVector();

            // If the column has zeros below the diagonal it is in the correct form. No need to compute reflector.
            if(col.maxAbs() > eps) {
                H = CMatrix.I(m);
                H.setSlice(Householder.getReflector(col), i, i);

                Q = Q.mult(H); // Apply Householder reflector to Q
                R = H.mult(R); // Apply Householder reflector to R
            }
        }
    }
}
