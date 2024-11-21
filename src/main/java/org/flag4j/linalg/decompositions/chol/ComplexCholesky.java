/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>An instance of this class allows for the computation of a Cholesky decomposition of a complex Hermitian
 * positive-definite matrix.</p>
 *
 * <p>Given a complex Hermitian positive-definite matrix A, the Cholesky decomposition will decompose it into
 * A=LL<sup>H</sup> where L is a lower triangular matrix and L<sup>H</sup> is the conjugate
 * transpose of L.</p>
 */
public class ComplexCholesky extends Cholesky<CMatrix> {

    /**
     * <p>Constructs a complex Cholesky decomposer.</p>
     *
     * <p>If you would like to enforce a check for hermitian symmetry at the time
     * of decomposition, see {@link #ComplexCholesky(boolean)}.</p>
     */
    public ComplexCholesky() {
        super(false);
    }


    /**
     * Constructs a complex Cholesky decomposer.
     *
     * @param checkPosDef flag indicating if the matrix to be decomposed should be explicitly checked to be hermitian ({@code true}).
     * If{@code false}, no check will be made and the matrix will be treated as if it were hermitian and only the lower half of the
     * matrix will be accessed
     */
    public ComplexCholesky(boolean checkPosDef) {
        super(checkPosDef);
    }


    /**
     * Decompose a matrix into A=LL<sup>H</sup> where L is a lower triangular matrix and L<sup>H</sup> is
     * the conjugate transpose of L.
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
            throw new IllegalArgumentException("Matrix is not Hermitian positive-definite.");
        } else {
            ValidateParameters.ensureSquareMatrix(src.shape);
        }

        L = new CMatrix(src.numRows);
        double posDefTolerance = Math.max(L.numRows*Flag4jConstants.EPS_F64, DEFAULT_POS_DEF_TOLERANCE);
        Complex128 sum;

        int lIndex1;
        int lIndex2;
        int lIndex3;

        for(int i=0; i<src.numCols; i++) {
            lIndex1 = i*L.numCols;

            for(int j=0; j<=i; j++) {
                sum = Complex128.ZERO;
                lIndex2 = j*L.numCols;
                lIndex3 = lIndex1 + j;

                for(int k=0; k<j; k++) {
                    sum = sum.add(L.entries[lIndex1 + k].mult(L.entries[lIndex2 + k].conj()));
                }

                if(i==j) {
                    Complex128 diag = src.entries[lIndex3].sub(sum);
                    if(diag.re <= 0 || diag.mag() <= posDefTolerance) {
                        throw new LinearAlgebraException("Matrix is not Hermitian positive-definite.");
                    }

                    L.entries[lIndex3] = diag.sqrt();
                } else {
                    if(!L.entries[j*(L.numCols + 1)].isZero()) {
                        L.entries[lIndex3] = (src.entries[lIndex3].sub(sum)).div((Complex128) L.entries[lIndex2 + j]);
                    }
                }
            }
        }

        return this;
    }
}
