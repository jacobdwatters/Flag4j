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

package com.flag4j.linalg.solvers;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.linalg.decompositions.ComplexLUDecomposition;
import com.flag4j.util.ParameterChecks;

import static com.flag4j.operations.dense.real.RealDenseDeterminant.detLU;


/**
 * Solver for solving a well determined system of linear equations in an exact sense using the
 * {@link com.flag4j.linalg.decompositions.LUDecomposition LU decomposition.}
 */
public class ComplexExactSolver extends ExactSolver<CMatrix, CVector> {

    private final ComplexForwardSolver forwardSolver;
    private final ComplexBackSolver backSolver;

    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    private static final double RANK_CONDITION = 1.0E-8;

    /**
     * Constructs an exact LU solver where the coefficient matrix is real dense.
     */
    public ComplexExactSolver() {
        super(new ComplexLUDecomposition());

        forwardSolver = new ComplexForwardSolver(true);
        backSolver = new ComplexBackSolver();
    }


    /**
     * Solves the linear system of equations given by {@code A*x=b} for {@code x}.
     *
     * @param A Coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     */
    @Override
    public CVector solve(CMatrix A, CVector b) {
        ParameterChecks.assertSquare(A.shape); // Ensure A is square.
        ParameterChecks.assertEquals(A.numCols, b.size); // b must have the same number of entries as columns in A.

        setUpSolver(A); // Compute LU decomposition and ensure the coefficient matrix is not singular.

        CVector y = forwardSolver.solve(L, P.mult(b));
        return backSolver.solve(U, y);
    }


    /**
     * Solves the set of linear system of equations given by {@code A*X=B} for the matrix {@code X} where
     * {@code A}, {@code B}, and {@code X} are matrices.
     *
     * @param A Coefficient matrix in the linear system.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=B}.
     */
    @Override
    public CMatrix solve(CMatrix A, CMatrix B) {
        ParameterChecks.assertSquare(A.shape); // Ensure A is square.
        ParameterChecks.assertEquals(A.numCols, B.numRows); // b must have the same number of entries as columns in A.

        setUpSolver(A); // Compute LU decomposition and ensure the coefficient matrix is not singular.

        CMatrix Y = forwardSolver.solve(L, P.mult(B));
        return backSolver.solve(U, Y);
    }


    /**
     * Computes the LU decomposition and checks to ensure the coefficient matrix is not singular.
     */
    private void setUpSolver(CMatrix A) {
        decompose(A); // Compute the decomposition of the coefficient matrix.

        double det = detLU(L, U).magAsDouble(); // Compute the determinant using the LU decomposition.

        if(det <= RANK_CONDITION*Math.max(A.numRows, A.numCols) || Double.isNaN(det)) {
            throw new SingularMatrixException("Could not solve system.");
        }
    }
}
