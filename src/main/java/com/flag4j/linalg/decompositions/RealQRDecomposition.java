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


import com.flag4j.Matrix;
import com.flag4j.Vector;


/**
 * Computes the QR decomposition for a real matrix.
 */
public final class RealQRDecomposition extends QRDecomposition<Matrix> {


    /**
     * Constructs a {@code QR} decomposer which computes the full {@code QR} decomposition.
     */
    public RealQRDecomposition() {
        super();
    }


    /**
     * Constructs a {@code QR} decomposer which computes either the full or reduced {@code QR} decomposition.
     * @param fullQR Flag for determining if the full {@code QR} decomposition should be used.
     *               If true, the full {@code QR} decomposition will be computed, if false,
     *               the reduced {@code QR} decomposition will be computed.
     */
    public RealQRDecomposition(boolean fullQR) {
        super(fullQR);
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. Not modified.
     */
    @Override
    public void decompose(Matrix src) {
        if(fullQR) {
            this.full(src);
        } else {
            this.reduced(src);
        }
    }


    /**
     * Computes the reduced QR decomposition on the src matrix.
     * @param A The source matrix to decompose.
     */
    private void reduced(Matrix A) {
        full(A); // First compute the full decomposition

        int k = Math.min(A.numRows, A.numCols);

        // Now reduce the decomposition
        Q = Q.getSlice(0, A.numRows, 0, k);
        R = R.getSlice(0, k, 0, A.numCols);
    }


    /**
     * Computes the full QR decomposition on the src matrix.
     * @param A The source matrix to decompose.
     */
    private void full(Matrix A) {
        R = new Matrix(A); // Initialize R to the values in A.
        int m = R.numRows, n = R.numCols;
        int stop = Math.min(n, m-1);

        Matrix H, col;

        // Initialize Q to the identity matrix.
        Q = Matrix.I(R.numRows);

        for(int i=0; i<stop; i++) {
            H = Matrix.I(m);
            col = R.getColBelow(i, i);

            if(!col.isZeros()) { // Then a householder transform must be applied
                H.setSlice(getHouseholder(col), i, i);

                System.out.println("H:\n" + H + "\n");

                Q = Q.mult(H); // Apply Householder reflector to Q
                R = H.mult(R); // Apply Householder reflector to R
            }
        }
    }


    /**
     * Computes the Householder reflector for the specified column vector.
     * @param col Column vector to compute Householder reflector for.
     * @return The Householder transformation matrix.
     */
    private Matrix getHouseholder(Matrix col) {
        Matrix H = Matrix.I(col.numRows);
        Vector v = col.toVector();

        double signedNorm = -Math.copySign(v.norm(), v.entries[0]);
        v = v.scalDiv(v.entries[0] + signedNorm);
        v.entries[0] = 1;

        // Create projection matrix
        Matrix P = v.outerProduct(v).scalMult(2/v.innerProduct(v));
        H.subEq(P);

        return H;
    }
}
