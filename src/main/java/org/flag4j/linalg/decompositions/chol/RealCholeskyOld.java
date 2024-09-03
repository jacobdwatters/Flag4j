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

package org.flag4j.linalg.decompositions.chol;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>An instance of this class allows for the computation of a CholeskyOld decomposition for a
 * real dense {@link MatrixOld matrix}.</p>
 *
 * <p>Given a symmetric positive-definite matrix {@code A}, the CholeskyOld decomposition will decompose it into
 * {@code A=LL<sup>T</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>T</sup>} is the
 * transpose of {@code L}.</p>
 */
@Deprecated
public final class RealCholeskyOld extends CholeskyOld<MatrixOld> {


    /**
     * Constructs a CholeskyOld decomposer. If you would like to enforce a check for symmetry at the time
     * of decomposition, see {@link #RealCholeskyOld(boolean)}.
     */
    public RealCholeskyOld() {
        super(false);
    }


    /**
     * Constructs a CholeskyOld decomposer.
     *
     * @param enforceSymmetric Flag indicating if the symmetry of the matrix to be decomposed should be explicitly checked (true).
     *                      If false, no check will be made and the matrix will be treated as if it were symmetric and only the
     *                      lower half of the matrix will be accessed.
     */
    public RealCholeskyOld(boolean enforceSymmetric) {
        super(enforceSymmetric);
    }


    /**
     * Decompose a matrix into {@code A=LL}<sup>T</sup> where {@code L} is a lower triangular matrix and {@code L}<sup>T</sup> is the
     * transpose of {@code L}.
     *
     * @param src The source matrix to decompose. Must be symmetric positive-definite. Note, symmetry will only be checked explicitly
     *            if {@link #RealCholeskyOld(boolean) enforceSymmetric} was set to true when this decomposer was
     *            instantiated.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code src} is not symmetric and {@link #RealCholeskyOld(boolean)
     * enforceSymmetric} was set to true when this decomposer was instantiated.
     * @throws LinearAlgebraException If this matrix is not positive-definite.
     */
    @Override
    public RealCholeskyOld decompose(MatrixOld src) {
        if(enforceHermitian && src.isSymmetric()) {
            throw new LinearAlgebraException("MatrixOld must be symmetric positive-definite.");
        } else {
            ParameterChecks.ensureSquareMatrix(src.shape);
        }

        L = new MatrixOld(src.numRows);
        double posDefTolerance = Math.max(L.numRows*Flag4jConstants.EPS_F64, DEFAULT_POS_DEF_TOLERANCE);
        double sum;

        int lIndex1;
        int lIndex2;
        int lIndex3;

        for(int i=0; i<src.numCols; i++) {
            lIndex1 = i*L.numCols;

            for(int j=0; j<=i; j++) {
                sum = 0;
                lIndex2 = j*L.numCols;
                lIndex3 = lIndex1 + j;

                for(int k=0; k<j; k++) {
                    sum += L.entries[lIndex1 + k]*L.entries[lIndex2 + k];
                }

                if(i==j) {
                    double diag = src.entries[lIndex3]-sum;
                    if(diag <= posDefTolerance) {
                        // Diagonal entries of L must be positive (non-zero) for original matrix to be positive-definite.
                        throw new LinearAlgebraException("MatrixOld is not symmetric positive-definite.");
                    }

                    L.entries[lIndex3] = Math.sqrt(diag);
                } else {
                    if(L.entries[j*(L.numCols + 1)] != 0) {
                        L.entries[lIndex3] = (src.entries[lIndex3]-sum)/L.entries[lIndex2 + j];
                    }
                }
            }
        }

        return this;
    }
}
