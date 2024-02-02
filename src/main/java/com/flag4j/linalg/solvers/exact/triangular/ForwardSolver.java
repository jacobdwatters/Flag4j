package com.flag4j.linalg.solvers.exact.triangular;

import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.exceptions.SingularMatrixException;
import com.flag4j.linalg.solvers.LinearSolver;
import com.flag4j.util.ParameterChecks;


/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector. That is, solves, {@code L*x=b} or {@code L*X=B} for the vector {@code x} or the
 * matrix {@code X} respectively where {@code L} is a lower triangular matrix.
 *
 * @param <T> Type of coefficient matrix.
 * @param <U> Vector type equivalent to the coefficient matrix.
 * @param <V> Type of the internal storage datastructures in the matrix and vector.
 */
public abstract class ForwardSolver<
        T extends MatrixMixin<T, ?, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, ?, ?, ?, ?, T, ?, ?>,
        V> implements LinearSolver<T, U> {

    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    protected static final double RANK_CONDITION = Math.ulp(1.0);
    /**
     * Flag indicating if lower-triangular matrices passed to this solver will be unit lower-triangular (true) or simply
     * lower-triangular (false).
     */
    protected final boolean isUnit;
    /**
     * Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular. If false, the matrix will
     * simply be assumed to be lower triangular.
     */
    protected final boolean enforceLower;
    /**
     * Storage for solution in solves which return a 00000000
     */
    T X;
    /**
     * Storage for solution in solves which return a vector.
     */
    U x;
    /**
     * Temporary storage for columns of the solution matrix. This can be used to improve cache performance when columns need to
     * be traveled.
     */
    protected V xCol;


    /**
     * Creates a solver for solving a lower-triangular system.
     * @param isUnit Flag indicating if coefficient matrices passed will be unit lower-triangular or simply lower-triangular in
     *               general.
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    protected ForwardSolver(boolean isUnit, boolean enforceLower) {
        this.isUnit = isUnit;
        this.enforceLower = enforceLower;
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

        if(enforceLower && !coeff.isTriL()) {
            throw new IllegalArgumentException("Expecting matrix L to be lower triangular.");
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
