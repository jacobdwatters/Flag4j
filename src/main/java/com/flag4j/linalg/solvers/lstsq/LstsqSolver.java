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

package com.flag4j.linalg.solvers.lstsq;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.core.MatrixMixin;
import com.flag4j.core.VectorMixin;
import com.flag4j.linalg.decompositions.qr.QRDecompositionOld;
import com.flag4j.linalg.solvers.LinearSolver;


// TODO: Add option to use SVD instead of QR (SVD should be default). It will be slower but has better numerical properties.

/**
 * This class solves a linear system of equations {@code Ax=b} in a least-squares sense. That is,
 * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
 * This is done using a {@link QRDecompositionOld}.
 */
public abstract class LstsqSolver<
        T extends MatrixMixin<T, T, ?, CMatrix, ?, U, U>,
        U extends VectorMixin<U, U, ?, CVector, ?, T, T, CMatrix>>
        implements LinearSolver<T, U> {

    /**
     * Solver for system with an upper triangular coefficient matrix.
     */
    protected final LinearSolver<T, U> backSolver;
    /**
     * Decomposer to compute the {@code QR} decomposition for using the least-squares solver.
     */
    protected final QRDecompositionOld<T, ?> qr;
    /**
     * {@code Q} The hermation transpose of the orthonormal matrix from the {@code QR} decomposition.
     */
    protected T Qh;
    /**
     * {@code R} The upper triangular matrix from the {@code QR} decomposition.
     */
    protected T R;

    /**
     * Constructs a least-squares solver with a specified decomposer to use in the {@code QR} decomposition.
     * @param qr The {@code QR} decomposer to use in the solver.
     * @param backSolver The solver to solve the upper triangular system resulting from the {@code QR} decomposition
     *                   which is equivalent to solving the normal equations
     */
    protected LstsqSolver(QRDecompositionOld<T, ?> qr, LinearSolver<T, U> backSolver) {
        this.qr = qr;
        this.backSolver = backSolver;
    }


    /**
     * Solves the linear system given by {@code Ax=b} in the least-squares sense.
     *
     * @param A Coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The least squares solution to {@code x} in the linear system {@code Ax=b}.
     */
    @Override
    public U solve(T A, U b) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, Qh.mult(b));
    }


    /**
     * Solves the set of linear system of equations given by {@code A*X=B} for the matrix {@code X} where
     * {@code A}, {@code B}, and {@code X} are matrices.
     *
     * @param A Coefficient matrix in the linear system.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=B}.
     */
    @Override
    public T solve(T A, T B) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, Qh.mult(B));
    }


    /**
     * Computes the {@code QR} decomposition for use in this solver.
     * @param A Coefficient matrix in the linear system to solve.
     */
    protected void decompose(T A) {
        qr.decompose(A);
        Qh = qr.getQ().H();
        R = qr.getR();
    }
}
