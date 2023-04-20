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

package com.flag4j.linalg.solvers;


import com.flag4j.core.MatrixBase;
import com.flag4j.core.VectorBase;
import com.flag4j.linalg.decompositions.QRDecomposition;


// TODO: Switch to SVD instead of QR. It will be slower but have better numerical properties.

/**
 * This class solves a linear system of equations {@code Ax=b} in a least-squares sense. That is,
 * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
 * This is done using a {@link QRDecomposition}.
 */
public abstract class LstsqSolver<T extends MatrixBase<?>,
        U extends VectorBase<?>, V extends VectorBase<?>> implements LinearSolver<T, U, V> {

    /**
     * Decomposer to compute the {@code QR} decomposition for using the least-squares solver.
     */
    protected final QRDecomposition<T> qr;
    /**
     * {@code Q} The orthonormal matrix from the {@code QR} decomposition.
     */
    protected T Q;
    /**
     * {@code R} The upper triangular matrix from the {@code QR} decomposition.
     */
    protected T R;

    /**
     * Constructs a least-squares solver with a specified decomposer to use in the {@code QR} decomposition.
     * @param qr The {@code QR} decomposer to use in the solver.
     */
    protected LstsqSolver(QRDecomposition<T> qr) {
        this.qr = qr;
    }


    /**
     * Computes the {@code QR} decomposition for use in this solver.
     * @param A Coefficient matrix in the linear system to solve.
     */
    protected void decompose(T A) {
        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
    }
}
