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
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.linalg.decompositions.lu.LUDecomposition;
import com.flag4j.linalg.decompositions.lu.RealLUDecomposition;
import com.flag4j.operations.dense.real.RealDenseDeterminant;


/**
 * Solver for solving a well determined system of linear equations in an exact sense using the
 * {@link LUDecomposition LU decomposition.}
 */
public class RealExactSolver extends ExactSolver<Matrix, Vector> {

    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    private static final double RANK_CONDITION = 1.0e-15;

    /**
     * Constructs an exact LU solver where the coefficient matrix is real dense.
     */
    public RealExactSolver() {
        super(new RealLUDecomposition(),
                new RealForwardSolver(true),
                new RealBackSolver()
        );
    }


    /**
     * Checks if the matrix is singular by computing the determinant using the LU decomposition assuming that
     * the LU decomposition produces a unit lower triangular matrix for {@code L}.
     * @throws SingularMatrixException If the matrix U contains a zero along the diagonal.
     */
    @Override
    protected void checkSingular() {
        double det = Math.abs(RealDenseDeterminant.detTri(upper));

        if(det <= RANK_CONDITION*Math.max(lower.numRows, upper.numCols) || Double.isNaN(det)) {
            throw new SingularMatrixException("Could not solve system.");
        }
    }


    /**
     * Permute the rows of a vector using the row permutation matrix from the LU decomposition.
     *
     * @param b Vector to permute the rows of.
     * @return A vector which is the result of applying the row permutation from the LU decomposition
     * to the vector {@code b}.
     */
    @Override
    protected Vector permuteRows(Vector b) {
        return rowPermute.leftMult(b);
//        return rowPermute.mult(b);
    }


    /**
     * Permute the rows of a matrix using the row permutation matrix from the LU decomposition.
     *
     * @param B matrix to permute the rows of.
     * @return A matrix which is the result of applying the row permutation from the LU decomposition
     * to the matrix {@code B}.
     */
    @Override
    protected Matrix permuteRows(Matrix B) {
        return rowPermute.leftMult(B);
//        return rowPermute.mult(B);
    }
}
