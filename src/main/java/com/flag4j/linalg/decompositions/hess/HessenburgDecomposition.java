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

import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.linalg.decompositions.Decomposition;
import com.flag4j.util.ParameterChecks;

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
public abstract class HessenburgDecomposition<
        T extends MatrixMixin<T, T, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, U, ?, ?, ?, T, T, ?>>
        implements Decomposition<T> {

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
     * Hessenburg decomposition of a square matrix. That is, for a square matrix
     * {@code A}, computes the decomposition {@code A=QHQ<sup>T</sup>} where {@code Q} is a unitary matrix and
     * {@code H} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues/vectors).</p>
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If the {@code src} matrix is not square.
     */
    @Override
    public HessenburgDecomposition<T, U> decompose(T src) {
        ParameterChecks.assertSquare(src.shape());

        // TODO: Add a decomposition for symmetric matrices.
        generalDecomposition(src);

        return this;
    }


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


    /**
     * Computes the Hessenburg decomposition of a general matrix.
     * @param src Matrix to compute the Hessenburg decomposition of.
     */
    protected void generalDecomposition(T src) {
        // Tolerance for considering a value zero when determining if a column of H is in the correct form.
        double tol = Math.ulp(1.0);

        H = src.copy(); // Storage for upper Hessenburg matrix
        T ref; // For storing Householder reflector
        U col; // Normal vector for Householder reflector computation.

        Q = computeQ ? initQ() : null;

        for(int k = 0; k<H.numRows()-2; k++) {
            col = H.getCol(k, k+1, H.numCols());

            // If the column is zeros, no need to compute reflector. It is already in the correct form.
            if(col.maxAbs() > tol) {
                ref = initRef(col); // Initialize a Householder reflector.

                // Apply Householder reflector to both sides of H.
                H.setSlice(
                        ref.mult(H.getSlice(k + 1, k+1+ref.numRows(), k, H.numCols())),
                        k + 1, k
                );
                H.setSlice(
                        H.getSlice(0, H.numRows(), k + 1, k + 1 + ref.numCols()).mult(ref.H()),
                        0, k + 1
                );


                if(computeQ) {
                    // Collect similarity transformations.
                    Q.setSlice(
                            Q.getSlice(0, Q.numRows(), k+1, k+1+ref.numCols()).mult(ref),
                            0, k + 1
                    );
                }
            }

            setZeros(k);
        }
    }


    /**
     * Creates a Householder reflector embedded in an identity matrix with the same size as {@code H}.
     * @param col Vector to compute Householder reflector for.
     * @return Householder reflector embedded in an identity matrix with the same size as {@code H}.
     */
    protected abstract T initRef(U col);


    /**
     * Initializes the unitary matrix {@code Q} in the Hessenburg decomposition.
     * @return The initial {@code Q} matrix in the Hessenburg decomposition.
     */
    protected abstract T initQ();


    /**
     * Sets the specified column below the first sub-diagonal to zero.
     * @param k Index of column to set values below the first sub-diagonal to zero.
     */
    protected abstract void setZeros(int k);
}
