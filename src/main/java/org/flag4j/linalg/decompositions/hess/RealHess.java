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


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.unitary.RealUnitaryDecomposition;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>Computes the Hessenberg decomposition of a real dense square matrix.
 * <p>The Hessenberg decomposition decomposes a given square matrix <b>A</b> into the product:
 * <pre>
 *     <b>A = QHQ<sup>T</sup></b></pre>
 * where <b>Q</b> is an orthogonal matrix and <b>H</b> is an upper Hessenberg matrix, which is similar to <b>A</b>
 * (i.e., it has the same eigenvalues).
 *
 * <p>A matrix <b>H</b> is in upper Hessenburg form if it is nearly upper triangular. Specifically, if <b>H</b> has
 * all zeros below the first sub-diagonal.
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each '&times;' may hold a different value:
 * <pre>
 *     [[ &times; &times; &times; &times; &times; ]
 *      [ &times; &times; &times; &times; &times; ]
 *      [ 0 &times; &times; &times; &times; ]
 *      [ 0 0 &times; &times; &times; ]
 *      [ 0 0 0 &times; &times; ]]</pre>
 *
 * <h2>Efficiency Considerations:</h2>
 * <ul>
 *     <li>If the orthogonal matrix <b>Q</b> is not required, setting {@code computeQ = false} in the constructor
 *     <em>may</em> improve performance.</li>
 *     <li>Support for in-place decomposition to reduce memory usage.</li>
 *     <li>Support for decomposition of matrix sub-blocks, enabling efficient eigenvalue computations.</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code RealHess}.</li>
 *     <li>Call {@link #decompose(Matrix)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getH()} and {@link #getQ()}.</li>
 * </ol>
 *
 * @see ComplexHess
 * @see #getH()
 * @see #getQ()
 * @see org.flag4j.linalg.decompositions.balance.RealBalancer
 */
public class RealHess extends RealUnitaryDecomposition {

    /**
     * <p>Creates a real Hessenburg decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros
     * below the first sub-diagonal. That is, reduce to an upper Hessenburg matrix.
     *
     * <p>By default, the orthogonal matrix <em>will</em> be computed. To specify if the orthogonal matrix should be computed, use
     * {@link #RealHess(boolean)}.
     *
     * @see #RealHess(boolean)
     * @see #RealHess(boolean, boolean)
     */
    public RealHess() {
        super(1);
    }


    /**
     * <p>Creates a real Hessenburg decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros
     * below the first sub-diagonal. That is, reduce to an upper Hessenburg matrix.
     *
     * @param computeQ Flag indicating if the orthogonal matrix in the Hessenburg decomposition should be computed. If it is not
     * needed, setting this to {@code false} <em>may</em> yield an increase in performance.
     * @see #RealHess()
     * @see #RealHess(boolean, boolean)
     */
    public RealHess(boolean computeQ) {
        super(1, computeQ);
    }


    /**
     * <p>Creates a real Hessenburg decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros
     * below the first sub-diagonal. That is, reduce to an upper Hessenburg matrix.
     *
     * @param computeQ Flag indicating if the orthogonal matrix in the Hessenburg decomposition should be computed. If it is not
     * needed, setting this to {@code false} <em>may</em> yield an increase in performance.
     * @param inPlace Flag indicating if the decomposition should be done in-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in place.</li>
     *     <li>If {@code false}, then the decomposition will be done out-of-place.</li>
     * </ul>
     *
     * @see #RealHess()
     * @see #RealHess(boolean)
     */
    public RealHess(boolean computeQ, boolean inPlace) {
        super(1, computeQ, inPlace);
    }


    /**
     * <p>Computes the Hessenberg decomposition of the specified matrix.
     * 
     * <p>Note, the computation of the orthogonal matrix <b>Q</b> in the decomposition is
     * deferred until {@link #getQ()} is explicitly called. This allows for efficient decompositions when <b>Q</b> is not needed.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     * @throws LinearAlgebraException If {@code src} is not a square matrix.
     */
    @Override
    public RealHess decompose(Matrix src) {
        ValidateParameters.ensureSquare(src.shape);
        super.decompose(src); // Compute the decomposition.
        return this;
    }


    /**
     * <p>Applies decomposition to the source matrix. Note, the computation of the orthogonal matrix <b>Q</b> in the decomposition is
     * deferred until {@link #getQ()} is explicitly called. This allows for efficient decompositions when <b>Q</b> is not needed.
     *
     * <p>This method can be used specify that only a sub-block within the full matrix needs to be
     * reduced. This is useful when you know that an upper and lower diagonal block of the matrix is already
     * in the correct form, and you only need to reduce an inner sub-block of the full matrix.
     * Most commonly this would be useful after balancing a matrix using
     * {@link org.flag4j.linalg.decompositions.balance.RealBalancer RealBalancer}, which results in the form
     * <pre>
     *      [ <b>T1</b>   <b>X</b>  <b>Y</b>  ]
     *      [  <b>0</b>   <b>B</b>  <b>Z</b>  ]
     *      [  <b>0</b>   <b>0</b>  <b>T2</b> ]</pre>
     * where <b>T1</b> and <b>T2</b> are in upper-triangular form. As such, only the <b>B</b> block needs to be reduced.
     * The staring row/column index of <b>B</b> is specified by {@code iLow} (inclusive) and the ending row/column
     * index is specified by {@code iHigh} (exclusive). It should be noted that the blocks <b>X</b> and <b>Z</b> will also be updated
     * during the reduction of <b>B</b> so the full matrix must still be passed.
     *
     * @param src The source matrix to decompose.
     * @param iLow Lower bound (inclusive) of the sub-matrix to reduce to upper Hessenburg form.
     * @param iHigh Upper bound (exclusive) of the sub-matrix to reduce to upper Hessenburg form.
     * @return A reference to this decomposer.
     * @throws LinearAlgebraException If {@code src} is not a square matrix.
     */
    public RealHess decompose(Matrix src, int iLow, int iHigh) {
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
     * Gets the upper Hessenburg matrix <b>H</b> from the Hessenburg decomposition.
     * @return The upper Hessenburg matrix <b>H</b> from the Hessenburg decomposition.
     */
    public Matrix getH() {
        return getUpper(new Matrix(numRows));
    }
}
