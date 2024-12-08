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

package org.flag4j.linalg.solvers;

import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;

/**
 * <p>Interface representing a solver for linear systems involving matrices and vectors. Implementations
 * of this interface provide methods to solve equations such as <i>Ax=b</i> and
 * <i>AX=B</i>, where <i>A</i>, <i>B</i>, and <i>X</i> are matrices, and <i>x</i> and <i>b</i> are vectors.
 *
 * <p>Solvers may compute exact solutions or approximate solutions in a least squares sense, depending on the properties of the system.
 *
 * @param <T> The type of matrices in the linear system, extending {@link MatrixMixin}.
 * @param <U> The type of vectors in the linear system, extending {@link VectorMixin}.
 */
public interface LinearMatrixSolver<T extends MatrixMixin<T, ?, U, ?>,
        U extends VectorMixin<U, T, ?, ?>> extends LinearSolver<T> {


    /**
     * Solves the linear system of equations <i>Ax=b</i> for the vector <i>x</i>.
     *
     * @param A The coefficient matrix <i>A</i> in the linear system.
     * @param b The constant vector in the linear system.
     * @return The solution vector <i>x</i> satisfying <i>Ax=b</i>.
     */
    U solve(T A, U b);


    /**
     * Solves the linear matrix equation <i>AX=B</i> for the matrix <i>X</i>.
     *
     * @param A The coefficient matrix in the linear system.
     * @param B The constant matrix in the linear system.
     * @return The solution matrix <i>X</i> satisfying <i>AX=B</i>.
     */
    @Override
    T solve(T A, T B);
}
