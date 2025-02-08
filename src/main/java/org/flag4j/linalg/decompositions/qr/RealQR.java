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
import org.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;
import org.flag4j.linalg.decompositions.unitary.RealUnitaryDecomposition;

/**
 * <p>Computes the QR decomposition of a real dense matrix.
 *
 * <p>The QR decomposition factorizes a given matrix <b>A</b> into the product of an orthogonal matrix <b>Q</b>
 * and an upper triangular matrix <b>R</b>, such that:
 * <pre>
 *     A = QR</pre>
 *
 * <p>The decomposition can be computed in either the <em>full</em> or <em>reduced</em> form:
 * <ul>
 *     <li><b>Reduced QR decomposition:</b> When {@code reduced = true}, the decomposition produces a compact form where
 *         <b>Q</b> has the same number of rows as <b>A</b> but only as many columns as the rank of <b>A</b>.</li>
 *     <li><b>Full QR decomposition:</b> When {@code reduced = false}, the decomposition produces a square orthogonal matrix
 *         <b>Q</b> with the same number of rows as <b>A</b>.</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code RealQR}.</li>
 *     <li>Call {@link #decompose(Matrix)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getQ()} and {@link #getR()}.</li>
 * </ol>
 *
 * @implNote This class extends {@link ComplexUnitaryDecomposition} and provides implementations for computing the
 * QR decomposition efficiently. The decomposition uses Householder transformations to iteratively
 * zero out sub-diagonal entries while maintaining numerical stability.
 *
 * @see RealUnitaryDecomposition
 * @see #getR()
 * @see #getQ()
 */
public class RealQR extends RealUnitaryDecomposition {

    /**
     * Flag indicating if the reduced or full decomposition should be computed.
     * <ul>
     *     <li>If {@code true}: the reduced decomposition will be computed.</li>
     *     <li>If {@code false}: the full decomposition will be computed.</li>
     * </ul>
     */
    protected final boolean reduced;


    /**
     * Creates a QR decomposer. This decomposer will compute the reduced QR decomposition.
     * @see #RealQR(boolean)
     */
    public RealQR() {
        super(0);
        this.reduced = true;
    }


    /**
     * Creates a QR decomposer to compute either the full or reduced QR decomposition.
     *
     * @param reduced Flag indicating if the reduced or full decomposition should be computed.
     * <ul>
     *     <li>If {@code true}: the reduced decomposition will be computed.</li>
     *     <li>If {@code false}: the full decomposition will be computed.</li>
     * </ul>
     */
    public RealQR(boolean reduced) {
        super(0);
        this.reduced = reduced;
    }


    /**
     * Computes the QR decomposition of the provided matrix.
     * @param src The matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public RealQR decompose(Matrix src) {
        super.decompose(src);
        return this;
    }


    /**
     * Creates and initializes the <b>Q</b> matrix to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected Matrix initQ() {
        int qCols = reduced ? minAxisSize : numRows; // Get Q in reduced form or not.
        return Matrix.I(numRows, qCols);
    }


    /**
     * Gets the upper triangular matrix <b>R</b> from the last decomposition. Same as {@link #getR()}.
     *
     * @return The upper triangular matrix from the last decomposition.
     */
    @Override
    public Matrix getUpper() {
        return getR();
    }


    /**
     * Gets the upper triangular matrix <b>R</b> from the QR decomposition.
     * @return The upper triangular matrix <b>R</b> from the QR decomposition.
     */
    public Matrix getR() {
        int rRows = reduced ? minAxisSize : numRows; // Get R in reduced form or not.
        return getUpper(new Matrix(rRows, numCols));
    }
}
