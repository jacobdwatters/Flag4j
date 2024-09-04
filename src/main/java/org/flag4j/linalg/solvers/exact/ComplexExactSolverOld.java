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

package org.flag4j.linalg.solvers.exact;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LUOld;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolverOld;
import org.flag4j.linalg.solvers.exact.triangular.ComplexForwardSolverOld;


/**
 * Solver for solving a well determined system of linear equations in an exact sense using the
 * {@link LUOld LUOld decomposition.}
 */
public class ComplexExactSolverOld extends ExactSolverOld<CMatrixOld, CVectorOld> {

    /**
     * Constructs an exact LUOld solver where the coefficient matrix is real dense.
     */
    public ComplexExactSolverOld() {
        super(new ComplexLU(),
                new ComplexForwardSolverOld(true),
                new ComplexBackSolverOld()
        );
    }


    /**
     * Permute the rows of a vector using the row permutation matrix from the LUOld decomposition.
     *
     * @param b VectorOld to permute the rows of.
     * @return A vector which is the result of applying the row permutation from the LUOld decomposition
     * to the vector {@code b}.
     */
    @Override
    protected CVectorOld permuteRows(CVectorOld b) {
        return rowPermute.leftMult(b);
    }


    /**
     * Permute the rows of a matrix using the row permutation matrix from the LUOld decomposition.
     *
     * @param B matrix to permute the rows of.
     * @return A matrix which is the result of applying the row permutation from the LUOld decomposition
     * to the matrix {@code B}.
     */
    @Override
    protected CMatrixOld permuteRows(CMatrixOld B) {
        return rowPermute.leftMult(B);
    }
}
