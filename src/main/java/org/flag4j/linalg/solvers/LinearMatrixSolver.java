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
 * This interface specifies methods which all linear matrix system solvers should implement.
 *
 * <p>Solvers may solve in an exact sense or in a least squares sense.</p>
 *
 * @param <T> Type of the matrices in the linear system.
 * @param <U> Type of the vectors in the linear system.
 */
public interface LinearMatrixSolver<T extends MatrixMixin<T, ?, U, ?, ?>,
        U extends VectorMixin<U, T, ?, ?>> extends LinearSolver<T> {


    /**
     * Solves the linear system of equations given by A*x=b for the vector x.
     * @param A Coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The solution to x in the linear system A*x=b.
     */
    U solve(T A, U b);


    /**
     * Solves the set of linear system of equations given by A*X=B for the matrix X where
     * A, B, and X are matrices.
     * @param A Coefficient matrix in the linear system.
     * @param B Matrix of constants in the linear system.
     * @return The solution to X in the linear system A*X=B.
     */
    @Override
    T solve(T A, T B);
}
