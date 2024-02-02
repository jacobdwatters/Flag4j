package com.flag4j.linalg.solvers.exact;

import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.linalg.solvers.LinearSolver;
import com.flag4j.util.ParameterChecks;

/**
 * Base class for solvers which solve a linear system of equations {@code U*x=b} or {@code U*X=B} where {@code U} is an upper
 * triangular matrix. This is solved in an exact sense.
 * @param <T> Type of matrix to decompose.
 * @param <U> Vector type equivalent of matrix.
 * @param <V> Type of internal storage for the matrix and vector.
 */
public abstract class BackSolver<
        T extends MatrixMixin<T, ?, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, ?, ?, ?, ?, T, ?, ?>,
        V>
        implements LinearSolver<T, U> {


    /**
     * For storing matrix results.
     */
    protected T X;
    /**
     * For storing vector results.
     */
    protected U x;
    /**
     * For temporary storage of matrix columns to help improve cache performance.
     */
    protected V xCol;
    /**
     * Flag indicating if determinant should be computed.
     */
    protected final boolean enforceTriU;
    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    protected static final double RANK_CONDITION = Math.ulp(1.0);


    /**
     * Creates a solver for solving linear systems for upper triangular coefficient matrices.
     * @param enforceTriU Flag indicating if an explicit check should be made that the coefficient matrix is upper triangular.
     */
    public BackSolver(boolean enforceTriU) {
        this.enforceTriU = enforceTriU;
    }


    /**
     * Ensures passed parameters are valid for the back solver.
     * @param coeff Coefficient matrix in the linear system.
     * @param constantRows Number of rows in the constant vector or matrix.
     * @throws IllegalArgumentException If coeff is not square,  {@code coeff.numRows()!=constantRows}, or if {@code enforceTriU} is
     * true and {@code coeff} is not upper triangular.
     */
    protected void checkParams(T coeff, int constantRows) {
        ParameterChecks.assertSquare(coeff.shape());
        ParameterChecks.assertEquals(coeff.numRows(), constantRows);

        if(enforceTriU && !coeff.isTriU()) {
            throw new IllegalArgumentException("Expecting matrix U to be upper triangular.");
        }
    }


    /**
     * Checks if the coefficient matrix is singular based on the computed determinant.
     * @param detAbs Absolute value of computed determinant.
     * @param numRows Number of rows in the coefficient matrix.
     * @param numCols Number of columns in the coefficient matrix.
     */
    protected void checkSingular(double detAbs, int numRows, int numCols) {
        if(detAbs <= RANK_CONDITION*Math.max(numRows, numCols) || Double.isNaN(detAbs)) {
            throw new SingularMatrixException("Could not solve.");
        }
    }
}
