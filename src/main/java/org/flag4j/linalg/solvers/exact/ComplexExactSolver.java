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

package org.flag4j.linalg.solvers.exact;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.ComplexForwardSolver;

/**
 * <p>Solves a well determined system of equations <b>Ax=b</b> or <b>AX=B</b> in an exact sense by using a
 * {@link ComplexLU LU decomposition}
 * where <b>A</b>, <b>B</b>, and <b>X</b> are matrices, and <b>x</b> and <b>b</b> are vectors.
 *
 * <p>If the system is not well determined, i.e. <b>A</b> is square and full rank, then use a
 * {@link org.flag4j.linalg.solvers.lstsq.ComplexLstsqSolver least-squares solver}.
 *
 * <h2>Usage:</h2>
 * <p>A single system may be solved by calling either {@link #solve(CMatrix, CVector)} or
 * {@link #solve(CMatrix, CVector)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form <b>Ax=b</b> or <b>AX=B</b>
 * for the same coefficient matrix <b>A</b> but numerous constant vectors/matrices <b>b</b> or <b>B</b>. To do this, the workflow
 * would be as follows:
 * <ol>
 *     <li>Create an instance of {@code ComplexExactSolver}.</li>
 *     <li>Call {@link #decompose(CMatrix) decompse(A)} once on the coefficient matrix <b>A</b>.</li>
 *     <li>Call {@link #solve(CVector) solve(b)} or {@link #solve(CMatrix) solve(B)} as many times as needed to solve each
 *     system for with the various <b>b</b> vectors and/or <b>B</b> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link #decompose(CMatrix) decompse(A)} will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link #solve(CMatrix, CVector)}</li>
 *     <li>{@link #solve(CMatrix, CMatrix)}</li>
 * </ul>
 *
 * <p>Specialized solvers are provided for inversion using {@link #solveIdentity(CMatrix)}. This should be preferred
 * over calling on of the other solve methods and providing an identity matrix explicitly.
 *
 * @param <T> The type of the coefficient matrix in the linear system.
 * @param <U> The type of vector in the linear system.
 */
public class ComplexExactSolver extends ExactSolver<CMatrix, CVector> {

    /**
     * Constructs an exact LU solver where the coefficient matrix is real dense.
     */
    public ComplexExactSolver() {
        super(new ComplexLU(),
                new ComplexForwardSolver(true, false),
                new ComplexBackSolver(false));
    }


    /**
     * Permute the rows of a vector using the row permutation matrix from the LU decomposition.
     *
     * @param b Vector to permute the rows of.
     * @return A vector which is the result of applying the row permutation from the LU decomposition
     * to the vector {@code b}.
     */
    @Override
    protected CVector permuteRows(CVector b) {
        return rowPermute.leftMult(b);
    }


    /**
     * Permute the rows of a matrix using the row permutation matrix from the LU decomposition.
     *
     * @param B matrix to permute the rows of.
     * @return A matrix which is the result of applying the row permutation from the LU decomposition
     * to the matrix {@code B}.
     */
    @Override
    protected CMatrix permuteRows(CMatrix B) {
        return rowPermute.leftMult(B);
    }
}
