/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.linalg.decompositions.unitary.RealUnitaryDecompositionOld;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>Computes the Hessenburg decomposition of a real dense square matrix. That is, for a square matrix
 * {@code A}, computes the decomposition {@code A=QHQ}<sup>T</sup> where {@code Q} is an orthogonal matrix and
 * {@code H} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues).</p>
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
@Deprecated
public class RealHessOld extends RealUnitaryDecompositionOld {

    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros below
     * the specified sub-diagonal.
     */
    public RealHessOld() {
        super(1);
    }


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros below
     * the specified sub-diagonal.
     */
    public RealHessOld(boolean computeQ) {
        super(1, computeQ);
    }


    /**
     * Applies decomposition to the source matrix. Note, the computation of the unitary matrix {@code Q} in the decomposition is
     * deferred until {@link #getQ()} is explicitly called. This allows for efficient decompositions when {@code Q} is not needed.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     * @throws LinearAlgebraException If {@code src} is not a square matrix.
     */
    @Override
    public RealHessOld decompose(MatrixOld src) {
        ParameterChecks.ensureSquare(src.shape);
        decomposeBase(src);
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected MatrixOld initQ() {
        return MatrixOld.I(numRows);
    }


    /**
     * Gets the upper Hessenburg matrix from the last decomposition. Same as {@link #getH()}
     *
     * @return The upper Hessenburg matrix from the last decomposition.
     */
    @Override
    public MatrixOld getUpper() {
        return getH();
    }


    /**
     * Gets the upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     * @return The upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     */
    public MatrixOld getH() {
        return getUpper(new MatrixOld(numRows));
    }
}
