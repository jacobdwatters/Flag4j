/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.linalg.decompositions.hess;


import org.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;
import org.flag4j.util.ValidateParameters;

/**
 * <p>Computes the Hessenburg decomposition of a complex dense square matrix. That is, for a square matrix
 * A, computes the decomposition A=QHQ<sup>H</sup> where Q is an unitary matrix and
 * H is a matrix in upper Hessenburg form and is similar to A (i.e. has the same eigenvalues).</p>
 *
 * <p>A matrix H is in upper Hessenburg form if it is nearly upper triangular. Specifically, if H has
 * all zeros below the first sub-diagonal.</p>
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each 'x' is a placeholder which may hold a different
 * value:
 * <pre>
 *     [[ x x x x x ]
 *      [ x x x x x ]
 *      [ 0 x x x x ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 * </p>
 */
public class ComplexHess extends ComplexUnitaryDecomposition {


    /**
     * <p>Creates a complex Hessenburg decomposer. This decomposer will compute the Hessenburg decomposition
     * for complex dense matrices.</p>
     *
     * <p>By default, the unitary matrix <i>will</i> be computed. To specify if the unitary matrix should be computed, use
     * {@link #ComplexHess(boolean)}.</p>
     *
     * @see #ComplexHess(boolean)
     */
    public ComplexHess() {
        super(1);
    }


    /**
     * <p>Creates a complex Hessenburg decomposer. This decomposer will compute the Hessenburg decomposition
     * for complex dense matrices.</p>
     *
     * @param computeQ Flag indicating if the unitary matrix in the Hessenburg decomposition should be computed. If it is not
     * needed, setting this to {@code false} <i>may</i> yield a slight increase in efficiency.
     * @see #ComplexHess()
     */
    public ComplexHess(boolean computeQ) {
        super(1, computeQ);
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexHess decompose(CMatrix src) {
        ValidateParameters.ensureSquare(src.shape);
        decomposeUnitary(src);
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected CMatrix initQ() {
        return CMatrix.I(numRows);
    }


    /**
     * Gets the upper Hessenburg matrix from the last decomposition. Same as {@link #getH()}
     *
     * @return The upper Hessenburg matrix from the last decomposition.
     */
    @Override
    public CMatrix getUpper() {
        return getH();
    }


    /**
     * Gets the upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     * @return The upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     */
    public CMatrix getH() {
        return getUpper(new CMatrix(numRows));
    }
}
