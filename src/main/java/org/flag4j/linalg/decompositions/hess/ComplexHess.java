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

package org.flag4j.linalg.decompositions.hess;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>Computes the Hessenburg decomposition of a complex dense square matrix. That is, for a square matrix
 * A, computes the decomposition A=QHQ<sup>H</sup> where Q is an unitary matrix and
 * H is a matrix in upper Hessenburg form and is similar to A (i.e. has the same eigenvalues).
 *
 * <p>A matrix H is in upper Hessenburg form if it is nearly upper triangular. Specifically, if H has
 * all zeros below the first sub-diagonal.
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each '&times;' is a placeholder which may hold a different
 * value:
 * <pre>
 *     [[ &times; &times; &times; &times; &times; ]
 *      [ &times; &times; &times; &times; &times; ]
 *      [ 0 &times; &times; &times; &times; ]
 *      [ 0 0 &times; &times; &times; ]
 *      [ 0 0 0 &times; &times; ]]</pre>
 * @see RealHess
 */
public class ComplexHess extends ComplexUnitaryDecomposition {


    /**
     * <p>Creates a complex Hessenburg decomposer. This decomposer will compute the Hessenburg decomposition
     * for complex dense matrices.
     *
     * <p>By default, the unitary matrix <em>will</em> be computed. To specify if the unitary matrix should be computed, use
     * {@link #ComplexHess(boolean)}.
     *
     * @see #ComplexHess(boolean)
     * @see #ComplexHess(boolean, boolean)
     */
    public ComplexHess() {
        super(1);
    }


    /**
     * <p>Creates a complex Hessenburg decomposer. This decomposer will compute the Hessenburg decomposition
     * for complex dense matrices.
     *
     * @param computeQ Flag indicating if the unitary matrix in the Hessenburg decomposition should be computed. If it is not
     * needed, setting this to {@code false} <em>may</em> yield a slight increase in efficiency.
     * @see #ComplexHess()
     * @see #ComplexHess(boolean, boolean)
     */
    public ComplexHess(boolean computeQ) {
        super(1, computeQ);
    }


    /**
     * <p>Creates a complex Hessenburg decomposer. This decomposer will compute the Hessenburg decomposition
     * for complex dense matrices.
     *
     * @param computeQ Flag indicating if the unitary matrix in the Hessenburg decomposition should be computed. If it is not
     * needed, setting this to {@code false} <i>may</i> yield a slight increase in efficiency.
     * @param inPlace Flag indicating if the decomposition should be done in-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in place.</li>
     *     <li>If {@code false}, then the decomposition will be done out-of-place.</li>
     * </ul>
     *
     * @see #ComplexHess()
     * @see #ComplexHess(boolean)
     */
    public ComplexHess(boolean computeQ, boolean inPlace) {
        super(1, computeQ, inPlace);
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexHess decompose(CMatrix src) {
        ValidateParameters.ensureSquare(src.shape);
        super.decompose(src);
        return this;
    }


    /**
     * <p>Applies decomposition to the source matrix. Note, the computation of the orthogonal matrix {@code Q} in the decomposition is
     * deferred until {@link #getQ()} is explicitly called. This allows for efficient decompositions when {@code Q} is not needed.
     *
     * <p>This method can be used specify that only a sub-block within the full matrix needs to be
     * reduced. This is useful when you know that an upper and lower diagonal block of the matrix is already
     * in the correct form, and you only need to reduce an inner sub-block of the full matrix.
     * Most commonly this would be useful after balancing a matrix using
     * {@link org.flag4j.linalg.decompositions.balance.ComplexBalancer ComplexBalancer}, which results in the form
     * <pre>
     *      [ <b>T1</b>   <b>X</b>  <b>Y</b>  ]
     *      [  <b>0</b>   <b>B</b>  <b>Z</b>  ]
     *      [  <b>0</b>   <b>0</b>  <b>T2</b> ]</pre>
     * where <b>T1</b> and <b>T2</b> are in upper-triangular form. As such, only the <b>B</b> block needs to be reduced.
     * The staring row/column index of <b>B</b> (inclusive) is specified by {@code iLow} and the ending row/column
     * index (exclusive) is specified by {@code iHigh}. It should be noted that the blocks <b>X</b> and <b>Z</b> will also be updated
     * during the reduction of <b>B</b> so the full matrix must still be passed.
     *
     * @param src The source matrix to decompose.
     * @param iLow Lower bound (inclusive) of the sub-matrix to reduce to upper Hessenburg form.
     * @param iHigh Upper bound (exclusive) of the sub-matrix to reduce to upper Hessenburg form.
     * @return A reference to this decomposer.
     * @throws LinearAlgebraException If {@code src} is not a square matrix.
     */
    public ComplexHess decompose(CMatrix src, int iLow, int iHigh) {
        ValidateParameters.ensureSquare(src.shape);
        super.decompose(src, iLow, iHigh); // Compute the decomposition.
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
