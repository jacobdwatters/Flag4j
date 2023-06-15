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
import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.linalg.decompositions.LUDecomposition;
import com.flag4j.util.ParameterChecks;

/**
 * <p>Solves a well determined system of equations {@code Ax=b} in an exact sense by using a {@code LU} decomposition.</p>
 * <p>If the system is not well determined, i.e. {@code A} is square and full rank, then use a
 * {@link LstsqSolver least-squares solver}.</p>
 */
public abstract class ExactSolver<
        T extends MatrixMixin<T, ?, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, ?, ?, ?, ?, T, ?, ?>>
        implements LinearSolver<T, U> {

    /**
     * Forward Solver for solving system with lower triangular coefficient matrix.
     */
    protected final LinearSolver<T, U> forwardSolver;
    /**
     * Backwards solver for solving system with upper triangular coefficient matrix.
     */
    protected final LinearSolver<T, U> backSolver;

    /**
     * Decomposer to compute {@code LU} decomposition.
     */
    protected final LUDecomposition<T> lu;
    /**
     * Unit lower triangular matrix in {@code} LU decomposition.
     */
    protected T lower;
    /**
     * Upper triangular matrix in {@code} LU decomposition.
     */
    protected T upper;
    /**
     * Row permutation matrix for {@code LU} decomposition.
     */
    protected Matrix rowPermute;

    /**
     * Constructs an exact LU solver with a specified {@code LU} decomposer.
     * @param lu {@code LU} decomposer to employ in solving the linear system.
     * @throws IllegalArgumentException If the {@code LU} decomposer does not use partial pivoting.
     */
    protected ExactSolver(LUDecomposition<T> lu, LinearSolver<T, U> forwardSolver, LinearSolver<T, U> backSolver) {
        if(lu.pivotFlag!=LUDecomposition.Pivoting.PARTIAL) {
            throw new IllegalArgumentException("LU solver must use partial pivoting but got " +
                    lu.pivotFlag.name() + ".");
        }

        this.lu = lu;
        this.forwardSolver = forwardSolver;
        this.backSolver = backSolver;
    }


    /**
     * Decomposes A using an {@link LUDecomposition LU decomposition}.
     * @param A Matrix to decompose.
     */
    protected void decompose(T A) {
        lu.decompose(A);
        lower = lu.getL();
        upper = lu.getU();
        rowPermute = lu.getP();
    }


    /**
     * Solves the linear system of equations given by {@code A*x=b} for the vector {@code x}. The system must be well
     * determined.
     *
     * @param A Coefficient matrix in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of entries in
     * {@code b}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public U solve(T A, U b) {
        ParameterChecks.assertSquare(A.shape()); // Ensure A is square.
        ParameterChecks.assertEquals(A.numCols(), b.size()); // b must have the same number of entries as columns in A.

        decompose(A); // Compute LU decomposition.
        checkSingular(); // Ensure the coefficient matrix is not singular using LU decomposition.

        U y = forwardSolver.solve(lower, permuteRows(b));
        return backSolver.solve(upper, y);
    }


    /**
     * Solves the set of linear system of equations given by {@code A*X=B} for the matrix {@code X}.
     *
     * @param A Coefficient matrix in the linear system.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=B}.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of rows in
     * {@code B}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public T solve(T A, T B) {
        ParameterChecks.assertSquare(A.shape()); // Ensure A is square.
        ParameterChecks.assertEquals(A.numCols(), B.numRows()); // b must have the same number of entries as columns in A.

        decompose(A); // Compute LU decomposition.
        checkSingular(); // Ensure the coefficient matrix is not singular using LU decomposition.

        T Y = forwardSolver.solve(lower, permuteRows(B));
        return backSolver.solve(upper, Y);
    }


    /**
     * Checks if the coefficient matrix is singular by computing the determinant using the LU decomposition assuming that
     * the LU decomposition produces a unit lower triangular matrix for {@code L}.
     * @throws SingularMatrixException If the coefficient matrix is singular.
     */
    protected abstract void checkSingular();


    /**
     * Permute the rows of a vector using the row permutation matrix from the LU decomposition.
     * @param b Vector to permute the rows of.
     * @return A vector which is the result of applying the row permutation from the LU decomposition
     * to the vector {@code b}.
     */
    protected abstract U permuteRows(U b);


    /**
     * Permute the rows of a matrix using the row permutation matrix from the LU decomposition.
     * @param B matrix to permute the rows of.
     * @return A matrix which is the result of applying the row permutation from the LU decomposition
     * to the matrix {@code B}.
     */
    protected abstract T permuteRows(T B);
}
