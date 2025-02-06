/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.unitary.RealUnitaryDecomposition;

/**
 * <p>Instances of this class compute the {@code QR} decomposition of a {@link Matrix real dense matrix}.
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into an orthogonal matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.
 *
 * <p>Much of this code has been adapted from the EJML library.
 */
public class RealQR extends RealUnitaryDecomposition {

    /**
     * Flag indicating if the reduced (true) or full (false) {@code QR} decomposition should be computed.
     */
    protected final boolean reduced;


    /**
     * Creates a {@code QR} decomposer. This decomposer will compute the reduced {@code QR} decomposition.
     * @see #RealQR(boolean)
     */
    public RealQR() {
        super(0);
        this.reduced = true;
    }


    /**
     * Creates a {@code QR} decomposer to compute either the full or reduced {@code QR} decomposition.
     *
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    public RealQR(boolean reduced) {
        super(0);
        this.reduced = reduced;
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public RealQR decompose(Matrix src) {
        super.decompose(src);
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected Matrix initQ() {
        int qCols = reduced ? minAxisSize : numRows; // Get Q in reduced form or not.
        return Matrix.I(numRows, qCols);
    }


    /**
     * Gets the upper triangular matrix {@code R} from the last decomposition. Same as {@link #getR()}.
     *
     * @return The upper triangular matrix from the last decomposition.
     */
    @Override
    public Matrix getUpper() {
        return getR();
    }


    /**
     * Gets the upper triangular matrix {@code R} from the {@code QR} decomposition.
     * @return The upper triangular matrix {@code R} from the {@code QR} decomposition.
     */
    public Matrix getR() {
        int rRows = reduced ? minAxisSize : numRows; // Get R in reduced form or not.
        return getUpper(new Matrix(rRows, numCols));
    }
}
