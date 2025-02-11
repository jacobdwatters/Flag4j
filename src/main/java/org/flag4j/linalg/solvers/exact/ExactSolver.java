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
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.SingularMatrixException;

import static org.flag4j.linalg.decompositions.lu.LU.Pivoting.PARTIAL;

/**
 * <p>Solves a well determined system of equations <b>Ax=b</b> or <b>AX=B</b> in an exact sense by using a LU decomposition
 * where <b>A</b>, <b>B</b>, and <b>X</b> are matrices, and <b>x</b> and <b>b</b> are vectors.
 *
 * <p>If the system is not well determined, i.e. <b>A</b> is square and full rank, then use a
 * {@link org.flag4j.linalg.solvers.lstsq.LstsqSolver least-squares solver}.
 *
 * <h3>Usage:</h3>
 * <p>A single system may be solved by calling either {@link #solve(MatrixMixin, VectorMixin)} or
 * {@link #solve(MatrixMixin, VectorMixin)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form <b>Ax=b</b> or <b>AX=B</b>
 * for the same coefficient matrix <b>A</b> but numerous constant vectors/matrices <b>b</b> or <b>B</b>. To do this, the workflow
 * would be as follows:
 * <ol>
 *     <li>Create a concrete instance of {@code LinearMatrixSolver}.</li>
 *     <li>Call {@link #decompose(MatrixMixin) decompse(A)} once on the coefficient matrix <b>A</b>.</li>
 *     <li>Call {@link #solve(VectorMixin) solve(b)} or {@link #solve(MatrixMixin) solve(B)} as many times as needed to solve each
 *     system for with the various <b>b</b> vectors and/or <b>B</b> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link #decompose(MatrixMixin) decompse(A)} will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
 *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
 * </ul>
 *
 * <p>Specialized solvers are provided for inversion using {@link #solveIdentity(MatrixMixin)}. This should be preferred
 * over calling on of the other solve methods and providing an identity matrix explicitly.
 *
 * @param <T> The type of the coefficient matrix in the linear system.
 * @param <U> The type of vector in the linear system.
 */
public abstract class ExactSolver<T extends MatrixMixin<T, ?, U, ?>,
        U extends VectorMixin<U, T, ?, ?>>
        implements LinearMatrixSolver<T, U> {

    /**
     * Forward Solver for solving system with lower triangular coefficient matrix.
     */
    protected final ForwardSolver<T, U, ?> forwardSolver;
    /**
     * Backwards solver for solving system with upper triangular coefficient matrix.
     */
    protected final BackSolver<T, U, ?> backSolver;

    /**
     * Decomposer to compute LU decomposition.
     */
    protected final LU<T> lu;
    /**
     * The unit-lower and upper triangular matrices from the LU decomposition stored in a single matrix.
     */
    protected T LU;
    /**
     * Row permutation matrix for LU decomposition.
     */
    protected PermutationMatrix rowPermute;

    /**
     * Constructs an exact LU solver with a specified LU decomposer.
     * @param lu LU decomposer to employ in solving the linear system.
     * @param forwardSolver Solver to use when solving <b>LY = b</b>.
     * @param backSolver Solver to use when solving <b>LY = b</b>.
     * @throws IllegalArgumentException If the LU decomposer does not use partial pivoting.
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
     * Decomposes a matrix <b>A</b> using an {@link LU LU decomposition}. This decomposition is then used by
     * {@link #solve(VectorMixin)} and {@link #solve(MatrixMixin)} to efficiently solve the systems
     * <b>Ax=b</b> and <b>AX=B</b> respectively.
     *
     * <p><b>Note</b>: Any subsequent call to {@link #solve(MatrixMixin, VectorMixin)} or {@link #solve(MatrixMixin, MatrixMixin)}
     * after a call to this method will override the coefficient matrix.
     *
     * <p>This is useful, and more efficient than {@link #solve(MatrixMixin, VectorMixin)} and
     * {@link #solve(MatrixMixin, MatrixMixin)}, if you need to solve multiple systems of this form
     * for the same <b>A</b> but numerous <b>b</b>'s or <b>B</b>'s that may not all be available at the same time.
     *
     * @param A Matrix to decompose.
     */
    public void decompose(T A) {
        ValidateParameters.ensureSquareMatrix(A.getShape()); // Ensure A is square.

        try {
            lu.decompose(A);
        } catch (LinearAlgebraException e) {
            // Throw an exception indicating why the system could not be solved.
            throw new SingularMatrixException(e.getMessage());
        }

        LU = lu.getLU();
        rowPermute = lu.getP();
    }


    /**
     * <p>Solves the linear system of equations given by <b>Ax=b</b> for the vector <b>x</b>. The system must be well
     * determined.
     * @param b Vector of constants, <b>b</b>, in the linear system.
     * @return The solution to <b>x</b> in the linear system <b>Ax=b</b>.
     * @throws IllegalStateException If no coefficient matrix has been specified for this solver by first calling one of the following:
     * <ul>
     *     <li>{@link #decompose(MatrixMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
     * </ul>
     */
    public U solve(U b) {
        if (LU == null) {
            throw new IllegalStateException("Coefficient matrix has not been specified for this solver." +
                    "\nMust call decompose(...) or a solve(...) which accepts a coefficient matrix first.");
        }

        U y = forwardSolver.solve(LU, permuteRows(b));
        return backSolver.solve(LU, y);
    }


    /**
     * <p>Solves the set of linear system of equations given by <b>AX=B</b> for the matrix <b>X</b>.
     *
     * @param B Matrix of constants, <b>B</b>, in the linear system.
     * @throws IllegalStateException If no coefficient matrix has been specified for this solver by first calling one of the following:
     * <ul>
     *     <li>{@link #decompose(MatrixMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
     * </ul>
     */
    public T solve(T B) {
        if (LU == null) {
            throw new IllegalStateException("Coefficient matrix has not been specified for this solver." +
                    "\nMust call decompose(...) or a solve(...) which accepts a coefficient matrix first.");
        }

        T Y = forwardSolver.solve(LU, permuteRows(B));
        return backSolver.solve(LU, Y);
    }


    /**
     * <p>Solves the linear system of equations given by <b>Ax=b</b> for the vector <b>x</b>. The system must be well
     * determined.
     *
     * <p><b>Note</b>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <b>A</b>, in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param b Vector of constants, <b>b</b>, in the linear system.
     * @return The solution to <b>x</b> in the linear system <b>Ax=b</b>.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of data in
     * {@code b}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public U solve(T A, U b) {
        ValidateParameters.ensureAllEqual(A.numCols(), b.length()); // b must have the same number of data as columns in A.

        decompose(A); // Compute LU decomposition.

        U y = forwardSolver.solve(LU, permuteRows(b));
        return backSolver.solve(LU, y);
    }


    /**
     * <p>Solves the set of linear system of equations given by <b>AX=B</b> for the matrix <b>X</b>.
     *
     * <p><b>Note</b>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <b>A</b>, in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param B Matrix of constants, <b>B</b>, in the linear system.
     * @return The solution to <b>x</b> in the linear system <b>AX=B</b>.
     * @throws IllegalArgumentException If the number of columns in {@code A} is not equal to the number of rows in
     * {@code B}.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    @Override
    public T solve(T A, T B) {
        ValidateParameters.ensureAllEqual(A.numCols(), B.numRows()); // b must have the same number of data as columns in A.

        decompose(A); // Compute LU decomposition.

        T Y = forwardSolver.solve(LU, permuteRows(B));
        return backSolver.solve(LU, Y);
    }


    /**
     * Solves the set of linear system of equations given by <b>AX=I</b> for the matrix <b>x</b> where <b>I</b>
     * is the identity matrix of the appropriate size. Thus, <b>X = A<sup>-1</sup></b> meaning this method computes the inverse of
     * <b>A</b>.
     *
     * <p>This method should be preferred over {@code solve(A, Matrix.I(A.shape))} or {@code solve(A, CMatrix.I(A.shape))} as it uses
     * specialized solvers that take advantage of the structure of the identity matrix.
     *
     * @param A Coefficient matrix in the linear system.
     * @return The solution to <b>x</b> in the linear system <b>AX=I</b>.
     * @throws IllegalArgumentException If {@code A} is not square.
     * @throws SingularMatrixException If {@code A} is singular.
     */
    public T solveIdentity(T A) {
        ValidateParameters.ensureSquareMatrix(A.getShape()); // Ensure A is square.

        decompose(A); // Compute LU decomposition.

        T Y = forwardSolver.solve(LU, lu.getP());
        return backSolver.solve(LU, Y);
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
