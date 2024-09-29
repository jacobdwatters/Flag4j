/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.linalg.solvers.exact.triangular;


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.linalg.solvers.LinearMatrixSolver;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.SingularMatrixException;

/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector. That is, solves, L*x=b or L*X=B for the vector x or the
 * matrix X respectively where L is a lower triangular matrix.
 *
 * @param <T> Type of coefficient matrix.
 * @param <U> Vector type equivalent to the coefficient matrix.
 * @param <V> Type of the internal storage datastructures in the matrix and vector.
 */
public abstract class ForwardSolver<T extends MatrixMixin<T, ?, U, ?, ?>, U extends VectorMixin<U, T, ?, ?>, V>
        implements LinearMatrixSolver<T, U> {

    /**
     * Threshold for determining if a determinant is to be considered zero when checking if the coefficient matrix is
     * full rank.
     */
    protected static final double RANK_CONDITION = Flag4jConstants.EPS_F64;
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
        ValidateParameters.ensureSquare(coeff.getShape());
        ValidateParameters.ensureEquals(coeff.numRows(), constantRows);

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
