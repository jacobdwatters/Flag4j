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


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.unitary.RealUnitaryDecomposition;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>Computes the Hessenburg decomposition of a real dense square matrix. That is, for a square matrix
 * A, computes the decomposition A=QHQ<sup>T</sup> where Q is an orthogonal matrix and
 * H is a matrix in upper Hessenburg form which is similar to A (i.e. has the same eigenvalues).</p>
 *
 * <p>A matrix H is in upper Hessenburg form if it is nearly upper triangular. Specifically, if H has
 * all zeros below the first sub-diagonal.</p>
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each 'x' may hold a different value:
 * <pre>
 *     [[ x x x x x ]
 *      [ x x x x x ]
 *      [ 0 x x x x ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 * </p>
 */
public class RealHess extends RealUnitaryDecomposition {

    /**
     * <p>Creates a real Hessenburg decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros
     * below the first sub-diagonal. That is, reduce to an upper Hessenburg matrix.</p>
     *
     * <p>By default, the orthogonal matrix <i>will</i> be computed. To specify if the orthogonal matrix should be computed, use
     * {@link #RealHess(boolean)}.</p>
     *
     * @see #RealHess(boolean)
     */
    public RealHess() {
        super(1);
    }


    /**
     * <p>Creates a real Hessenburg decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros
     * below the first sub-diagonal. That is, reduce to an upper Hessenburg matrix.</p>
     *
     * @param computeQ Flag indicating if the orthogonal matrix in the Hessenburg decomposition should be computed. If it is not
     * needed, setting this to {@code false} <i>may</i> yield a slight increase in efficiency.
     * @see #RealHess()
     */
    public RealHess(boolean computeQ) {
        super(1, computeQ);
    }


    /**
     * Applies decomposition to the source matrix. Note, the computation of the orthogonal matrix {@code Q} in the decomposition is
     * deferred until {@link #getQ()} is explicitly called. This allows for efficient decompositions when {@code Q} is not needed.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     * @throws LinearAlgebraException If {@code src} is not a square matrix.
     */
    @Override
    public RealHess decompose(Matrix src) {
        ValidateParameters.ensureSquare(src.shape);
        decomposeUnitary(src); // Compute the decomposition.
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected Matrix initQ() {
        return Matrix.I(numRows);
    }


    /**
     * Gets the upper Hessenburg matrix from the last decomposition. Same as {@link #getH()}
     *
     * @return The upper Hessenburg matrix from the last decomposition.
     */
    @Override
    public Matrix getUpper() {
        return getH();
    }


    /**
     * Gets the upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     * @return The upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     */
    public Matrix getH() {
        return getUpper(new Matrix(numRows));
    }
}
