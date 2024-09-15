package org.flag4j.linalg.solvers.exact;


import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.decompositions.lu.RealLU;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;
import org.flag4j.linalg.solvers.exact.triangular.RealForwardSolver;

/**
 * Solver for solving a well determined system of linear equations in an exact sense using the
 * {@link LU LU decomposition.}
 */
public class RealExactSolver extends ExactSolver<Matrix, Vector> {

    /**
     * Constructs an exact LU solver where the coefficient matrix is real dense.
     */
    public RealExactSolver() {
        super(new RealLU(), new RealForwardSolver(true), new RealBackSolver());
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
    }
}
