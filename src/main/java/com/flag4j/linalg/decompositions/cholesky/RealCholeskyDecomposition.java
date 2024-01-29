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

package com.flag4j.linalg.decompositions.cholesky;

import com.flag4j.Matrix;
import com.flag4j.linalg.PositiveDefiniteness;
import com.flag4j.util.ParameterChecks;


/**
 * <p>An instance of this class allows for the computation of a Cholesky decomposition for a
 * real dense {@link Matrix matrix}.</p>
 *
 * <p>Given a symmetric positive-definite matrix {@code A}, the Cholesky decomposition will decompose it into
 * {@code A=LL<sup>T</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>T</sup>} is the
 * transpose of {@code L}.</p>
 */
public final class RealCholeskyDecomposition extends CholeskyDecomposition<Matrix> {


    /**
     * Constructs a Cholesky decomposer. If you would like to enforce a check for positive definiteness at the time
     * of decomposition, see {@link #RealCholeskyDecomposition(boolean)}. However, note that this will be significantly
     * slower.
     */
    public RealCholeskyDecomposition() {
        super(false);
    }


    /**
     * Constructs a Cholesky decomposer.
     *
     * @param enforcePosDef Flag indicating if the symmetric positive definiteness of the matrix should be checked
     *                    before decomposing.
     *                    If true, a check for positive definiteness will be done before the matrix is decomposed. If
     *                    it is not, an error will be thrown. If false, no check will be made and the matrix will be
     *                    assumed to be positive definite. <b>WARNING: </b> Checking positive definiteness can
     *                    be very computationally expensive.
     */
    public RealCholeskyDecomposition(boolean enforcePosDef) {
        super(enforcePosDef);
    }


    /**
     * Gets the transpose of the {@code L} matrix computed by the Cholesky decomposition {@code A=LL<sup>T</sup>}.
     *
     * @return The transpose of the {@code L} matrix from the Cholesky decomposition {@code A=LL<sup>T</sup>}.
     */
    @Override
    public Matrix getLH() {
        return L.T();
    }


    /**
     * Decompose a matrix into {@code A=LL<sup>T</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>T</sup>} is the
     * transpose of {@code L}.
     *
     * @param src The source matrix to decompose. Must be symmetric positive-definite.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code src} is not symmetric positive-definite.
     */
    @Override
    public RealCholeskyDecomposition decompose(Matrix src) {
        if(enforcePosDef && !(src.isSymmetric() && PositiveDefiniteness.isPosDef(src))) {
            throw new IllegalArgumentException("Matrix must be positive definite.");
        } else if(!enforcePosDef) {
            ParameterChecks.assertSquare(src.shape);
        }

        L = new Matrix(src.numRows);
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
                    L.entries[lIndex3] = Math.sqrt(src.entries[lIndex3]-sum);
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
