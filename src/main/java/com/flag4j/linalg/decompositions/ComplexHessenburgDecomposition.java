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
import com.flag4j.linalg.transformations.Householder;
import com.flag4j.util.ParameterChecks;

/**
 * <p>Computes the Hessenburg decomposition of a complex dense square matrix. That is, for a square matrix
 * {@code A}, computes the decomposition {@code A=QHQ<sup>H</sup>} where {@code Q} is a unitary matrix and
 * {@code H} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues/vectors).</p>
 *
 * <p>A matrix {@code B} is in upper Hessenburg form if it is nearly upper triangular. Specifically, if {@code B} has
 * all zeros below the first sub-diagonal.</p>
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each {@code x} may hold a different value:
 * <pre>
 *     [[ x x x x x ]
 *      [ x x x x x ]
 *      [ 0 x x x x ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 * </p>
 */
public final class ComplexHessenburgDecomposition extends HessenburgDecomposition<CMatrix> {

    /**
     * Constructs a {@link ComplexHessenburgDecomposition Hessenburg decomposer} for real dense matrices.
     * This decomposer will compute the unitary matrix {@code Q} corresponding to {@code A=QHQ<sup>H</sup>}. To
     * create a Hessenburg decomposer which <b>does not</b> compute {@code Q}, see {@link #ComplexHessenburgDecomposition(boolean)}.
     */
    public ComplexHessenburgDecomposition() {
        super(true);
    }


    /**
     * Constructs a {@link RealHessenburgDecomposition Hessenburg decomposer} for real dense matrices. The
     * @param computeQ Flag for computing {@code Q} in the decomposition corresponding to {@code A=QHQ<sup>H</sup>}.<br>
     *                 - If true, {@code Q} will be computed.<br>
     *                 - If false, {@code Q} will <b>not</b> be computed. This may give a performance increase if
     *                 {@code Q} is not needed.
     */
    public ComplexHessenburgDecomposition(boolean computeQ) {
        super(computeQ);
    }


    /**
     * Hessenburg decomposition of a real dense square matrix. That is, for a square matrix
     * {@code A}, computes the decomposition {@code A=QHQ<sup>H</sup>} where {@code Q} is a unitary matrix and
     * {@code B} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues/vectors).</p>
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     */
    @Override
    public ComplexHessenburgDecomposition decompose(CMatrix src) {
        ParameterChecks.assertSquare(src.shape);

        H = src.copy(); // Storage for upper Hessenburg matrix
        CMatrix ref; // For storing Householder reflector
        CVector col; // Normal vector for Householder reflector computation.

        if(computeQ) {
            Q = CMatrix.I(this.H.numRows); // Storage for unitary matrix in the decomposition.
        } else {
            Q = null;
        }

        for(int k = 0; k< this.H.numRows-2; k++) {
            col = this.H.getColBelow(k+1, k).toVector();

            if(!col.isZeros()) { // If the column is zeros, no need to compute reflector. It is already in the correct form.
                ref = CMatrix.I(this.H.numRows);
                ref.setSlice(Householder.getReflector(col), k+1, k+1);

                H = ref.mult(H).mult(ref.H()); // Apply Householder reflector to both sides of B.

                if(computeQ) {
                    Q = Q.mult(ref); // Apply Householder reflector to Q.
                }
            }
        }

        return this;
    }
}
