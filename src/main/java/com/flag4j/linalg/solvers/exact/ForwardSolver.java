package com.flag4j.linalg.solvers.exact;

import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.linalg.solvers.LinearSolver;

import java.lang.reflect.Array;


/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector. That is, solves, {@code L*x=b} or {@code L*X=B} for the vector {@code x} or the
 * matrix {@code X} respectively where {@code L} is a lower triangular matrix.
 */
public abstract class ForwardSolver<
        T extends MatrixMixin<T, ?, ?, ?, ?, U, ?>,
        U extends VectorMixin<U, ?, ?, ?, ?, T, ?, ?>,
        V extends Array> implements LinearSolver<T, U> {

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
     * Creates a solver for solving a lower-triangular system.
     * @param isUnit Flag indicating if coefficient matrices passed will be unit lower-triangular or simply lower-triangular in
     *               general.
     * @param enforceLower Flag indicating if an explicit check should be made that the coefficient matrix is lower triangular.
     */
    protected ForwardSolver(boolean isUnit, boolean enforceLower) {
        this.isUnit = isUnit;
        this.enforceLower = enforceLower;
    }
}
