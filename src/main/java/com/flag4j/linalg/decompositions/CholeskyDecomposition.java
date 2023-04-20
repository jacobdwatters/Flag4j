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
 * <p>This abstract class specifies methods for computing the Cholesky decomposition of a positive-definite matrix.</p>
 *
 * <p>Given a hermation positive-definite matrix {@code A}, the Cholesky decomposition will decompose it into
 * {@code A=LL<sup>*</sup>} where {@code L} is a lower triangular matrix and {@code L<sup>*</sup>} is the conjugate
 * transpose of {@code L}.</p>
 *
 * <p>If {@code A} is a real valued symmetric positive-definite matrix, then the decomposition simplifies to
 * {@code A=LL<sup>T</sup>}.</p>
 *
 * @param <T> The type of matrix to compute the Cholesky decomposition of.
 */
public abstract class CholeskyDecomposition<T extends MatrixBase<?>> implements Decomposition<T> {

    /**
     * The lower triangular matrix resulting from the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     */
    protected T L;


    /**
     * Gets the {@code L} matrix computed by the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     * @return The {@code L} matrix from the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     */
    public T getL() {
        return L;
    }


    /**
     * Gets the conjugate transpose of the {@code L} matrix computed by the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     * @return The conjugate transpose of the {@code L} matrix from the Cholesky decomposition {@code A=LL<sup>*</sup>}.
     */
    public abstract T getLH();
}
