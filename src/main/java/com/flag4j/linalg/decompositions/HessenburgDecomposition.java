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

import com.flag4j.core.MatrixBase;


/**
 * <p>This abstract class specifies methods for computing the Hessenburg decomposition of a square matrix. That is, for a square matrix
 * {@code A}, computes the decomposition {@code A=QHQ<sup>H</sup>} where {@code Q} is a unitary matrix and
 * {@code H} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues/vectors).</p>
 *
 * <p>A matrix {@code H} is in upper Hessenburg form if it is nearly upper triangular. Specifically, if {@code H} has
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
public abstract class HessenburgDecomposition<T extends MatrixBase<?, ?, ?, ?, ?, ?, ?>> implements Decomposition<T> {

    /**
     * A flag for determining if {@code Q} should be computed in the Hessenburg decomposition.
     */
    protected boolean computeQ;

    /**
     * Storage for the {@code Q} matrix in the Hessenburg decomposition corresponding to {@code A=QHQ<sup>H</sup>}.
     */
    protected T Q;
    /**
     * Storage for the {@code H} matrix in the Hessenburg decomposition corresponding to {@code A=QHQ<sup>H</sup>}.
     */
    protected T H;


    /**
     * Constructs a Hessenburg decomposer which specifies if the unitary matrix {@code Q} in the decomposition should
     * be computed.
     * @param computeQ Flag for determining if the {@code Q} in the decomposition should be computed.<br>
     *                 - If true, the unitary {@code Q} matrix will be computed.<br>
     *                 - If false, the unitary {@code Q} matrix will not be computed which may give a performance
     *                 increase if it is not needed.
     */
    protected HessenburgDecomposition(boolean computeQ) {
        this.computeQ = computeQ;
    }


    /**
     * Gets the unitary {@code Q} matrix from the Hessenburg decomposition corresponding to {@code A=QHQ<sup>*</sup>}.
     * @return The unitary {@code Q} matrix from the Hessenburg decomposition corresponding to {@code A=QHQ<sup>*</sup>}.
     */
    public T getQ() {
        return Q;
    }


    /**
     * Gets the upper Hessenburg {@code H} matrix from the Hessenburg decomposition corresponding to {@code A=QHQ<sup>H</sup>}.
     * @return The upper Hessenburg {@code H} matrix from the Hessenburg decomposition corresponding to {@code A=QHQ<sup>H</sup>}.
     */
    public T getH() {
        return H;
    }
}
