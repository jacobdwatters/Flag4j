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
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.PositiveDefiniteness;
import com.flag4j.util.ParameterChecks;


/**
 * <p>This abstract class specifies methods for computing the Cholesky decomposition of a hermation
 * positive-definite matrix.</p>
 *
 * <p>Given a hermation positive-definite matrix {@code A}, the Cholesky decomposition will decompose it into
 * {@code A=LL<sup>*</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>*</sup>} is the conjugate
 * transpose of {@code L}.</p>
 */
public final class ComplexCholeskyDecomposition extends CholeskyDecomposition<CMatrix> {


    /**
     * Constructs a Cholesky decomposer. If you would like to enforce a check for positive definiteness at the time
     * of decomposition, see {@link #ComplexCholeskyDecomposition(boolean)}. However, note that this may have a
     * significant negative impact on performance.
     * slower.
     */
    public ComplexCholeskyDecomposition() {
        super(false);
    }


    /**
     * Constructs a Cholesky decomposer.
     *
     * @param checkPosDef Flag indicating if the positive definiteness of the matrix should be checked before decomposing.
     *                    If true, a check for positive definiteness will be done before the matrix is decomposed. If
     *                    it is not, an error will be thrown. If false, no check will be made and the matrix will be
     *                    assumed to be positive definite.
     */
    public ComplexCholeskyDecomposition(boolean checkPosDef) {
        super(checkPosDef);
    }


    /**
     * Decompose a matrix into {@code A=LL<sup>*</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>*</sup>} is the conjugate
     * transpose of {@code L}.
     *
     * @param src The source matrix to decompose. Must be hermation positive-definite.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code src} is not hermation positive-definite.
     */
    @Override
    public ComplexCholeskyDecomposition decompose(CMatrix src) {
        if(checkPosDef && !PositiveDefiniteness.isPosDef(src)) {
            throw new IllegalArgumentException("Matrix must be positive definite.");
        } else if(!checkPosDef) {
            ParameterChecks.assertSquare(src.shape);
        }

        L = new CMatrix(src.numRows);
        CNumber sum;

        int lIndex1;
        int lIndex2;
        int lIndex3;

        for(int i=0; i<src.numCols; i++) {
            lIndex1 = i*L.numCols;

            for(int j=0; j<=i; j++) {
                sum = new CNumber();
                lIndex2 = j*L.numCols;
                lIndex3 = lIndex1 + j;

                for(int k=0; k<j; k++) {
                    sum.addEq(L.entries[lIndex1 + k].mult(L.entries[lIndex2 + k].conj()));
                }

                if(i==j) {
                    L.entries[lIndex3] = CNumber.sqrt(src.entries[lIndex3].sub(sum));
                } else {
                    if(!L.entries[j*(L.numCols + 1)].equals(CNumber.ZERO)) {
                        L.entries[lIndex3] = (src.entries[lIndex3].sub(sum)).div(L.entries[lIndex2 + j]);
                    }
                }
            }
        }

        return this;
    }


    /**
     * Gets the conjugate transpose of the {@code L} matrix computed by the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     *
     * @return The conjugate transpose of the {@code L} matrix from the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     */
    @Override
    public CMatrix getLH() {
        return L.H();
    }
}
