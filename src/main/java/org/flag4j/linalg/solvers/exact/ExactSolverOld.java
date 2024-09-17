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


import org.flag4j.arrays_old.sparse.PermutationMatrixOld;
import org.flag4j.core_old.MatrixMixin;
import org.flag4j.core_old.VectorMixin;
import org.flag4j.linalg.decompositions.lu.LUOld;
import org.flag4j.linalg.solvers.LinearSolverOld;
import org.flag4j.linalg.solvers.lstsq.LstsqSolverOld;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * <p>Solves a well determined system of equations {@code Ax=b} in an exact sense by using a {@code LUOld} decomposition.</p>
 * <p>If the system is not well determined, i.e. {@code A} is square and full rank, then use a
 * {@link LstsqSolverOld least-squares solver}.</p>
 */
@Deprecated
public abstract class ExactSolverOld<
        T extends MatrixMixin<T, ?, ?, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, ?, ?, ?, ?, T, ?, ?>>
        implements LinearSolverOld<T, U> {

    /**
     * Forward Solver for solving system with lower triangular coefficient matrix.
     */
    protected final LinearSolverOld<T, U> forwardSolver;
    /**
     * Backwards solver for solving system with upper triangular coefficient matrix.
     */
    protected final LinearSolverOld<T, U> backSolver;

    /**
     * Decomposer to compute {@code LUOld} decomposition.
     */
    protected final LUOld<T> lu;
    /**
     * The unit-lower and upper triangular matrices from the {@code LUOld} decomposition stored in a single matrix.
     */
    protected T LU;
    /**
     * Row permutation matrix for {@code LUOld} decomposition.
     */
    protected PermutationMatrixOld rowPermute;

    /**
     * Constructs an exact LUOld solver with a specified {@code LUOld} decomposer.
     * @param lu {@code LUOld} decomposer to employ in solving the linear system.
     * @throws IllegalArgumentException If the {@code LUOld} decomposer does not use partial pivoting.
     */
    protected ExactSolverOld(LUOld<T> lu, LinearSolverOld<T, U> forwardSolver, LinearSolverOld<T, U> backSolver) {
        if(lu.pivotFlag!= LUOld.Pivoting.PARTIAL) {
            throw new IllegalArgumentException("LUOld solver must use partial pivoting but got " +
                    lu.pivotFlag.name() + ".");
        }

        this.lu = lu;
        this.forwardSolver = forwardSolver;
        this.backSolver = backSolver;
    }


    /**
     * Decomposes A using an {@link LUOld LUOld decomposition}.
     * @param A MatrixOld to decompose.
     */
    protected void decompose(T A) {
        lu.decompose(A);
        LU = lu.getLU();
        rowPermute = lu.getP();
    }


    /**
     * Solves the linear system of equations given by {@code A*x=b} for the vector {@code x}. The system must be well
     * determined.
     *
     * @param A Coefficient matrix in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param b VectorOld of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of entries in
     * {@code b}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public U solve(T A, U b) {
        ParameterChecks.ensureSquareMatrix(A.shape()); // Ensure A is square.
        ParameterChecks.ensureEquals(A.numCols(), b.size()); // b must have the same number of entries as columns in A.

        decompose(A); // Compute LUOld decomposition.

        U y = forwardSolver.solve(LU, permuteRows(b));
        return backSolver.solve(LU, y); // If A is singular, then U will be singular, and it will be discovered here.
    }


    /**
     * Solves the set of linear system of equations given by {@code A*X=B} for the matrix {@code X}.
     *
     * @param A Coefficient matrix in the linear system.
     * @param B MatrixOld of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=B}.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of rows in
     * {@code B}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public T solve(T A, T B) {
        ParameterChecks.ensureSquareMatrix(A.shape()); // Ensure A is square.
        ParameterChecks.ensureEquals(A.numCols(), B.numRows()); // b must have the same number of entries as columns in A.

        decompose(A); // Compute LUOld decomposition.

        T Y = forwardSolver.solve(LU, permuteRows(B));
        return backSolver.solve(LU, Y); // If A is singular, it will be discovered in the back solve.
    }


    /**
     * Permute the rows of a vector using the row permutation matrix from the LUOld decomposition.
     * @param b VectorOld to permute the rows of.
     * @return A vector which is the result of applying the row permutation from the LUOld decomposition
     * to the vector {@code b}.
     */
    protected abstract U permuteRows(U b);


    /**
     * Permute the rows of a matrix using the row permutation matrix from the LUOld decomposition.
     * @param B matrix to permute the rows of.
     * @return A matrix which is the result of applying the row permutation from the LUOld decomposition
     * to the matrix {@code B}.
     */
    protected abstract T permuteRows(T B);
}