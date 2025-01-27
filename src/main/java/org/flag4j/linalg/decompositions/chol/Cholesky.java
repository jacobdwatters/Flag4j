/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.linalg.decompositions.Decomposition;

/**
 * <p>This abstract class specifies methods for computing the Cholesky decomposition of a positive-definite matrix.
 *
 * <p>The Cholesky decomposition is essentially a special case of the {@link org.flag4j.linalg.decompositions.lu.LU LU
 * decomposition} for Hermitian positive-definite matrices where the upper triangular matrix is the Hermitian transpose
 * of the lower triangular matrix.
 *
 * <p>Given a Hermitian positive-definite matrix A, the Cholesky decomposition will decompose it into
 * A=LL<sup>H</sup> where L is a lower triangular matrix and L<sup>H</sup> is the conjugate
 * transpose of L.
 *
 * <p>If A is a real valued symmetric positive-definite matrix, then the decomposition simplifies to
 * A=LL<sup>T</sup>.
 *
 * @param <T> The type of matrix to compute the Cholesky decomposition of.
 */
public abstract class Cholesky<T extends MatrixMixin<T, ?, ?, ?>> implements Decomposition<T> {

    /**
     * Default tolerance for considering a value along the diagonal of L to be non-positive.
     */
    protected static final double DEFAULT_POS_DEF_TOLERANCE = 1.0e-10;

    /**
     * Flag indicating if the matrix to be decomposed should be explicitly checked to be Hermitian (true). If false, no check
     * will be made and the matrix will be treated as if it were Hermitian and only the lower half of the matrix will be accessed.
     */
    final boolean enforceHermitian;

    /**
     * Constructs a Cholesky decomposer.
     * @param enforceHermitian Flag indicating if the matrix to be decomposed should be explicitly checked to be Hermitian
     * ({@code true}). If {@code false}, no check will be made and the matrix will be treated as if it were Hermitian and only the
     * lower half of the matrix will be accessed.
     */
    protected Cholesky(boolean enforceHermitian) {
        this.enforceHermitian = enforceHermitian;
    }


    /**
     * The lower triangular matrix, L, resulting from the Cholesky decomposition A=LL<sup>H</sup>.
     */
    protected T L;


    /**
     * Gets the L matrix computed by the Cholesky decomposition A=LL<sup>H</sup>.
     * @return The {@code L} matrix from the Cholesky decomposition A=LL<sup>H</sup>.
     */
    public T getL() {
        return L;
    }


    /**
     * Gets the L<sup>H</sup> matrix computed by the Cholesky decomposition A=LL<sup>H</sup>.
     * @return The L<sup>H</sup> matrix from the Cholesky decomposition A=LL<sup>H</sup>.
     */
    public T getLH() {
        return L.H();
    }
}
