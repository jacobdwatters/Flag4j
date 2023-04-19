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

import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.linalg.decompositions.RealLUDecomposition;
import com.flag4j.util.ParameterChecks;

import static com.flag4j.operations.dense.real.RealDenseDeterminant.detLU;


/**
 * Solver for solving a well determined system of linear equations in an exact sense using the
 * {@link com.flag4j.linalg.decompositions.LUDecomposition LU decomposition.}
 */
public class MatrixLUSolver extends LUSolver<Matrix, Vector, Vector> {

    private MatrixForwardSolver forwardSolver;
    private MatrixBackSolver backSolver;

    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    private static final double RANK_THRESHOLD = 1.0e-8;

    /**
     * Constructs an exact LU solver where the coefficient matrix is real dense.
     */
    public MatrixLUSolver() {
        super(new RealLUDecomposition());

        forwardSolver = new MatrixForwardSolver();
        backSolver = new MatrixBackSolver();
    }


    /**
     * Solves the linear system of equations given by {@code A*x=b} for {@code x}.
     *
     * @param A Coefficient matrix in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of entries in
     * {@code b}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws IllegalArgumentException If {@code A} is not full rank.
     */
    @Override
    public Vector solve(Matrix A, Vector b) {
        ParameterChecks.assertSquare(A.shape); // Ensure A is square.
        ParameterChecks.assertEquals(A.numCols, b.size); // b must have the same number of entries as columns in A.
        decompose(A); // Compute the decomposition of the coefficient matrix.

        double det = Math.abs(detLU(L, U));

        if(det <= RANK_THRESHOLD || Double.isNaN(det)) {
            throw new IllegalArgumentException("Matrix expected to have full rank but did not.");
        }

        Vector y = forwardSolver.solve(L, P.mult(b).toVector());
        return backSolver.solve(U, y);
    }
}
