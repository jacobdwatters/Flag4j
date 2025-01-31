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


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.solvers.LinearMatrixSolver;
import org.flag4j.linalg.solvers.exact.triangular.BackSolver;
import org.flag4j.linalg.solvers.exact.triangular.ForwardSolver;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

import static org.flag4j.linalg.decompositions.lu.LU.Pivoting.PARTIAL;

// TODO: Javadoc needs updated. This can solve Ax=b or AX=B.
/**
 * <p>Solves a well determined system of equations Ax=b in an exact sense by using a LU decomposition.
 * <p>If the system is not well determined, i.e. {@code A} is square and full rank, then use a
 * {@link org.flag4j.linalg.solvers.lstsq.LstsqSolver least-squares solver}.
 *
 * @param <T> The type of the coefficient matrix in the linear system.
 * @param <U> The type of vector in the linear system.
 */
public abstract class ExactSolver<T extends MatrixMixin<T, ?, U, ?>,
        U extends VectorMixin<U, T, ?, ?>>
        implements LinearMatrixSolver<T, U> {

    // TODO: A huge benefit of using a decomposition based solver is that you can compute the decomposition (LU or other) once
    //      in O(n^3) then use that to solve Ax=b using any b in O(n^2). So, someone may want to keep a solver around and solve
    //      multiple Ax=b problems for different b at different times. Solvers should allow for that. You should be able to
    //      create an instance of this class tied to a coefficient matrix so this can be done. All solvers (exact and lstsq)
    //      should be converted to use this API.


    /**
     * Forward Solver for solving system with lower triangular coefficient matrix.
     */
    protected final ForwardSolver<T, U, ?> forwardSolver;
    /**
     * Backwards solver for solving system with upper triangular coefficient matrix.
     */
    protected final BackSolver<T, U, ?> backSolver;

    /**
     * Decomposer to compute {@code LU} decomposition.
     */
    protected final LU<T> lu;
    /**
     * The unit-lower and upper triangular matrices from the {@code LU} decomposition stored in a single matrix.
     */
    protected T LU;
    /**
     * Row permutation matrix for {@code LU} decomposition.
     */
    protected PermutationMatrix rowPermute;

    /**
     * Constructs an exact LU solver with a specified {@code LU} decomposer.
     * @param lu {@code LU} decomposer to employ in solving the linear system.
     * @param forwardSolver Solver to use when solving <b>LY = b</b>.
     * @param backSolver Solver to use when solving <b>LY = b</b>.
     * @throws IllegalArgumentException If the {@code LU} decomposer does not use partial pivoting.
     */
    protected ExactSolver(LU<T> lu, ForwardSolver<T, U, ?> forwardSolver, BackSolver<T, U, ?> backSolver) {
        if(lu.pivotFlag!= PARTIAL) {
            throw new IllegalArgumentException("LU solver must use partial pivoting but got " +
                    lu.pivotFlag.name() + ".");
        }

        this.lu = lu;
        this.forwardSolver = forwardSolver;
        this.backSolver = backSolver;
    }


    /**
     * Decomposes A using an {@link LU LU decomposition}.
     * @param A Matrix to decompose.
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
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of data in
     * {@code b}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public U solve(T A, U b) {
        ValidateParameters.ensureSquareMatrix(A.getShape()); // Ensure A is square.
        ValidateParameters.ensureAllEqual(A.numCols(), b.length()); // b must have the same number of data as columns in A.

        decompose(A); // Compute LU decomposition.

        U y = forwardSolver.solve(LU, permuteRows(b));
        return backSolver.solve(LU, y); // If A is singular, then U will be singular, and it will be discovered here.
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
        ValidateParameters.ensureSquareMatrix(A.getShape()); // Ensure A is square.
        ValidateParameters.ensureAllEqual(A.numCols(), B.numRows()); // b must have the same number of data as columns in A.

        decompose(A); // Compute LU decomposition.

        T Y = forwardSolver.solve(LU, permuteRows(B));
        return backSolver.solve(LU, Y); // If A is singular, it will be discovered in the back solve.
    }


    /**
     * Solves the set of linear system of equations given by {@code A*X=I} for the matrix {@code X} where I is the identity matrix
     * of the appropriate size.
     *
     * @param A Coefficient matrix in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=I}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    public T solveIdentity(T A) {
        ValidateParameters.ensureSquareMatrix(A.getShape()); // Ensure A is square.

        decompose(A); // Compute LU decomposition.

        T Y = forwardSolver.solve(LU, lu.getP());
        return backSolver.solve(LU, Y); // If A is singular, it will be discovered in the back solve.
    }


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
