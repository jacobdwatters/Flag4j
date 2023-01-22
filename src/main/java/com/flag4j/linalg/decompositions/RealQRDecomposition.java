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

/**
 * Computes the QR decomposition for a real matrix.
 */
public class RealQRDecomposition extends QRDecomposition<Matrix> {


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. Not modified.
     */
    @Override
    public void decompose(Matrix src) {
        if(fullQR) {
            this.full(src.copy());
        } else {
            this.reduced(src.copy());
        }
    }


    /**
     * Computes the reduced QR decomposition on the src matrix.
     * @param A The source matrix to decompose. Modified.
     */
    private void reduced(Matrix A) {
        int m = A.numRows, n = A.numCols;
        int stop = Math.min(n, m-1);
        Matrix H, x;

        // Initialize Q to the identity matrix.
        Q = Matrix.I(A.numRows);

        for(int i=0; i<stop; i++) {
            H = Matrix.I(m);
            H.setSlice(getHouseHolder(A.getCol(i)), i, i);
            Q = Q.mult(H); // Update Q
            A = A.mult(Q);
        }

    }


    /**
     * Computes the full QR decomposition on the src matrix.
     * @param A The source matrix to decompose. Modified.
     */
    private void full(Matrix A) {
        // TODO:
    }


    /**
     * Computes the Householder reflector for the specified column.
     * @param col Column to compute Householder reflector for.
     * @return The Householder transformation matrix.
     */
    private Matrix getHouseHolder(Matrix col) {
        Matrix H = Matrix.I(col.numRows);
        // TODO: Compute rest of Householder reflector.
        return H;
    }
}
