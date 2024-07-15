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

package org.flag4j.linalg.decompositions.qr;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;


/**
 * <p>Instances of this class compute the {@code QR} decomposition of a complex dense matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into a unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 *
 * <p>Much of this code has been adapted from the EJML library.</p>
 */
public class ComplexQR extends ComplexUnitaryDecomposition {

    /**
     * Flag indicating if the reduced (true) or full (false) {@code QR} decomposition should be computed.
     */
    protected final boolean reduced;

    /**
     * Creates a {@code QR} decomposer. This decomposer will compute the reduced {@code QR} decomposition.
     * @see #ComplexQR(boolean)
     */
    public ComplexQR() {
        super(0);
        this.reduced = true;
    }


    /**
     * Creates a {@code QR} decomposer to compute either the full or reduced {@code QR} decomposition.
     *
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    public ComplexQR(boolean reduced) {
        super(0);
        this.reduced = reduced;
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexQR decompose(CMatrix src) {
        decomposeBase(src);
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected CMatrix initQ() {
        int qCols = reduced ? minAxisSize : numRows; // Get Q in reduced form or not.
        return CMatrix.I(numRows, qCols);
    }


    /**
     * Gets the upper triangular matrix {@code R} from the last decomposition.
     *
     * @return The upper triangular matrix {@code R} from the last decomposition.
     */
    @Override
    public CMatrix getUpper() {
        return getR();
    }


    /**
     * Gets the upper triangular matrix {@code R} from the {@code QR} decomposition.
     * @return The upper triangular matrix {@code R} from the {@code QR} decomposition.
     */
    public CMatrix getR() {
        int rRows = reduced ? minAxisSize : numRows; // Get R in reduced form or not.
        return getUpper(new CMatrix(rRows, numCols));
    }
}
