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

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>This abstract class specifies methods for computing the Cholesky decomposition of a hermitian
 * positive-definite matrix.</p>
 *
 * <p>Given a hermitian positive-definite matrix {@code A}, the Cholesky decomposition will decompose it into
 * {@code A=LL<sup>*</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>*</sup>} is the conjugate
 * transpose of {@code L}.</p>
 */
public final class ComplexCholesky extends Cholesky<CMatrix> {


    /**
     * <p>Constructs a Cholesky decomposer.</p>
     *
     * <p>If you would like to enforce a check for hermitian symmetry at the time
     * of decomposition, see {@link #ComplexCholesky(boolean)}.</p>
     */
    public ComplexCholesky() {
        super(false);
    }


    /**
     * Constructs a Cholesky decomposer.
     *
     * @param checkPosDef flag indicating if the matrix to be decomposed should be explicitly checked to be hermitian (true). If
     *                    false, no check will be made and the matrix will be treated as if it were hermitian and only the lower
     *                    half of the matrix will be accessed
     */
    public ComplexCholesky(boolean checkPosDef) {
        super(checkPosDef);
    }


    /**
     * Decompose a matrix into {@code A=LL}<sup>H</sup> where {@code L} is a lower triangular matrix and {@code L}<sup>H</sup> is
     * the conjugate transpose of {@code L}.
     *
     * @param src The source matrix to decompose. Must be hermitian positive-definite.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code src} is not symmetric and {@link #ComplexCholesky(boolean)
     * enforceSymmetric} was set to true when this decomposer was instantiated.
     * @throws LinearAlgebraException If {@code src} is not positive-definite.
     */
    @Override
    public ComplexCholesky decompose(CMatrix src) {
        if(enforceHermitian && src.isHermitian()) {
            throw new IllegalArgumentException("Matrix must be positive-definite.");
        } else {
            ParameterChecks.assertSquareMatrix(src.shape);
        }

        L = new CMatrix(src.numRows);
        double posDefTolerance = Math.max(L.numRows*Flag4jConstants.EPS_F64, DEFAULT_POS_DEF_TOLERANCE);
        CNumber sum;

        int lIndex1;
        int lIndex2;
        int lIndex3;

        for(int i=0; i<src.numCols; i++) {
            lIndex1 = i*L.numCols;

            for(int j=0; j<=i; j++) {
                sum = CNumber.ZERO;
                lIndex2 = j*L.numCols;
                lIndex3 = lIndex1 + j;

                for(int k=0; k<j; k++) {
                    sum = sum.add(L.entries[lIndex1 + k].mult(L.entries[lIndex2 + k].conj()));
                }

                if(i==j) {
                    CNumber diag = src.entries[lIndex3].sub(sum);
                    if(diag.re <= 0 || diag.mag() <= posDefTolerance) {
                        throw new LinearAlgebraException("Matrix is not symmetric positive-definite.");
                    }

                    L.entries[lIndex3] = CNumber.sqrt(diag);
                } else {
                    if(!L.entries[j*(L.numCols + 1)].equals(0)) {
                        L.entries[lIndex3] = (src.entries[lIndex3].sub(sum)).div(L.entries[lIndex2 + j]);
                    }
                }
            }
        }

        return this;
    }
}
