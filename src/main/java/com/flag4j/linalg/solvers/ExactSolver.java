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


import com.flag4j.Matrix;
import com.flag4j.core.MatrixMixin;
import com.flag4j.linalg.decompositions.LUDecomposition;

/**
 * <p>Solves a well determined system of equations {@code Ax=b} in an exact sense by using a {@code LU} decomposition.</p>
 * <p>If the system is not well determined, i.e. {@code A} is square and full rank, then use a
 * {@link LstsqSolver least-squares solver}.</p>
 */
public abstract class ExactSolver<T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?>, U, V>
        implements LinearSolver<T, U, V> {

    /**
     * Decomposer to compute {@code LU} decomposition.
     */
    protected final LUDecomposition<T> lu;
    /**
     * Unit lower triangular matrix in {@code} LU decomposition.
     */
    protected T L;
    /**
     * Upper triangular matrix in {@code} LU decomposition.
     */
    protected T U;
    /**
     * Row permutation matrix for {@code LU} decomposition.
     */
    protected Matrix P;

    /**
     * Constructs an exact LU solver with a specified {@code LU} decomposer.
     * @param lu {@code LU} decomposer to employ in solving the linear system.
     * @throws IllegalArgumentException If the {@code LU} decomposer does not use partial pivoting.
     */
    protected ExactSolver(LUDecomposition<T> lu) {
        if(lu.pivotFlag!=LUDecomposition.Pivoting.PARTIAL) {
            throw new IllegalArgumentException("LU solver must use partial pivoting but got " +
                    lu.pivotFlag.name() + ".");
        }

        this.lu = lu;
    }


    /**
     * Decomposes A using an {@link LUDecomposition LU decomposition}.
     * @param A Matrix to decompose.
     */
    protected void decompose(T A) {
        lu.decompose(A);
        L = lu.getL();
        U = lu.getU();
        P = lu.getP();
    }
}
