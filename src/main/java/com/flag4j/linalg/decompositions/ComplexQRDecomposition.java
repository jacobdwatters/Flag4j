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
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.transformations.Householder;


/**
 * <p>Instances of this class compute the {@code QR} decomposition of a complex dense matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into an unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 */
public final class ComplexQRDecomposition extends QRDecomposition<CMatrix, CVector> {

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
     * Sets the specified column to zeros below the principle diagonal.
     * @param idx Index of the column for which to set entries below principle diagonal to zero.
     */
    @Override
    protected void setZeros(int idx) {
        for(int i=idx+1; i<R.numRows; i++) {
            R.entries[i*R.numCols + idx] = new CNumber();
        }
    }


    /**
     * Initializes the {@code Q} and {@code R} matrices.
     * @param src Matrix to decompose.
     */
    @Override
    protected void initQR(CMatrix src) {
        R = new CMatrix(src); // Initialize R to the values in src.
        Q = CMatrix.I(R.numRows); // Initialize Q to the identity matrix.
    }


    /**
     * Initializes a Householder reflector.
     *
     * @param col Vector to compute householder reflector for.
     * @param i   Row and column index to set slice of identity matrix as the Householder reflector.
     */
    @Override
    protected CMatrix initH(CVector col, int i) {
        CMatrix H = CMatrix.I(R.numRows);
        return H.setSlice(Householder.getReflector(col), i, i);
    }
}
