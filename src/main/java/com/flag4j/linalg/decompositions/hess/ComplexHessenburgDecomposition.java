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

package com.flag4j.linalg.decompositions.hess;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.transformations.Householder;

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
public final class ComplexHessenburgDecomposition extends HessenburgDecomposition<CMatrix, CVector> {

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
     * Creates a Householder reflector embedded in an identity matrix with the same size as {@code H}.
     *
     * @param col Vector to compute Householder reflector for.
     * @return Householder reflector embedded in an identity matrix with the same size as {@code H}.
     */
    @Override
    protected CMatrix initRef(CVector col) {
        return Householder.getReflector(col);
    }


    /**
     * Initializes the unitary matrix {@code Q} in the Hessenburg decomposition.
     *
     * @return The initial {@code Q} matrix in the Hessenburg decomposition.
     */
    @Override
    protected CMatrix initQ() {
        return CMatrix.I(this.H.numRows);
    }


    /**
     * Sets the specified column below the first sub-diagonal to zero.
     *
     * @param k Index of column to set values below the first sub-diagonal to zero.
     */
    @Override
    protected void setZeros(int k) {
        for(int i=k+2; i<H.numRows; i++) {
            H.entries[i*H.numCols + k] = new CNumber();
        }
    }
}