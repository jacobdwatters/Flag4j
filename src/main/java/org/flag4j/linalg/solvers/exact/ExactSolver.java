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
 * <p>Solves a well determined system of equations <strong>Ax = b</strong> or <strong>AX = B</strong> in an exact sense.
 * <p>If the system is not well determined, i.e. <strong>A</strong> is not square or not full rank, then use a
 * {@link org.flag4j.linalg.solvers.lstsq.LstsqSolver least-squares solver}.
 *
 * <h2>Usage:</h2>
 * <p>A single system may be solved by calling either {@link #solve(MatrixMixin, VectorMixin)} or
 * {@link #solve(MatrixMixin, VectorMixin)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form <strong>Ax = b</strong> or <strong>AX = B</strong>
 * for the same coefficient matrix <strong>A</strong> but numerous constant vectors/matrices <strong>b</strong> or <strong>B</strong>. To do this, the workflow
 * would be as follows:
 * <ol>
 *     <li>Create a concrete instance of {@code LinearMatrixSolver}.</li>
 *     <li>Call {@link #decompose(MatrixMixin) decompse(A)} once on the coefficient matrix <strong>A</strong>.</li>
 *     <li>Call {@link #solve(VectorMixin) solve(b)} or {@link #solve(MatrixMixin) solve(B)} as many times as needed to solve each
 *     system for with the various <strong>b</strong> vectors and/or <strong>B</strong> matrices. </li>
 * </ol>
 *
 * <strong>Note:</strong> Any call made to one of the following methods after a call to {@link #decompose(MatrixMixin) decompse(A)} will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
 *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
 * </ul>
 *
 * <p>Specialized solvers are provided for inversion using {@link #solveIdentity(MatrixMixin)}. This should be preferred
 * over calling on of the other solve methods and providing an identity matrix explicitly.
 * 
 * <h2>Implementation Notes:</h2>
 * The
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
     * @param forwardSolver Solver to use when solving <strong>LY = b</strong>.
     * @param backSolver Solver to use when solving <strong>LY = b</strong>.
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
     * Decomposes a matrix <strong>A</strong> using an {@link LU LU decomposition}. This decomposition is then used by
     * {@link #solve(VectorMixin)} and {@link #solve(MatrixMixin)} to efficiently solve the systems
     * <strong>Ax = b</strong> and <strong>AX = B</strong> respectively.
     *
     * <p><strong>Note</strong>: Any subsequent call to {@link #solve(MatrixMixin, VectorMixin)} or {@link #solve(MatrixMixin, MatrixMixin)}
     * after a call to this method will override the coefficient matrix.
     *
     * <p>This is useful, and more efficient than {@link #solve(MatrixMixin, VectorMixin)} and
     * {@link #solve(MatrixMixin, MatrixMixin)}, if you need to solve multiple systems of this form
     * for the same <strong>A</strong> but numerous <strong>b</strong>'s or <strong>B</strong>'s that may not all be available at the same time.
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
     * <p>Solves the linear system of equations given by <strong>Ax = b</strong> for the vector <strong>x</strong>. The system must be well
     * determined.
     * @param b Vector of constants, <strong>b</strong>, in the linear system.
     * @return The solution to <strong>x</strong> in the linear system <strong>Ax = b</strong> for the last <strong>A</strong> passed to
     * {@link #decompose(MatrixMixin)}.
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
     * <p>Solves the set of linear system of equations given by <strong>AX = B</strong> for the matrix <strong>X</strong>.
     *
     * @param B Matrix of constants, <strong>B</strong>, in the linear system.
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
     * <p>Solves the linear system of equations given by <strong>Ax = b</strong> for the vector <strong>x</strong>. The system must be well
     * determined.
     *
     * <p><strong>Note</strong>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <strong>A</strong>, in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param b Vector of constants, <strong>b</strong>, in the linear system.
     * @return The solution to <strong>x</strong> in the linear system <strong>Ax = b</strong>.
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
     * <p>Solves the set of linear system of equations given by <strong>AX = B</strong> for the matrix <strong>X</strong>.
     *
     * <p><strong>Note</strong>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <strong>A</strong>, in the linear system. Must be square and have full rank
     *          (i.e. all rows, or equivalently columns, must be linearly independent).
     * @param B Matrix of constants, <strong>B</strong>, in the linear system.
     * @return The solution to <strong>x</strong> in the linear system <strong>AX = B</strong>.
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
     * Solves the set of linear system of equations given by <strong>AX=I</strong> for the matrix <strong>x</strong> where <strong>I</strong>
     * is the identity matrix of the appropriate size. Thus, <strong>X = A<sup>-1</sup></strong> meaning this method computes the inverse of
     * <strong>A</strong>.
     *
     * <p>This method should be preferred over {@code solve(A, Matrix.I(A.shape))} or {@code solve(A, CMatrix.I(A.shape))} as it uses
     * specialized solvers that take advantage of the structure of the identity matrix.
     *
     * @param A Coefficient matrix in the linear system.
     * @return The solution to <strong>x</strong> in the linear system <strong>AX=I</strong>.
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
