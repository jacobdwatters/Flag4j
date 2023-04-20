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


/**
 * This interface specifies methods which all linear system solvers should implement. Solvers
 * may solve in an exact sense or in a least squares sense.
 * @param <T> Type of the matrix containing the coefficients of the linear system.
 * @param <U> Type of the vector containing the constants in the linear system.
 * @param <V> Type of the vector to hold the solution to the linear system.
 */
public interface LinearSolver<T extends MatrixBase<?>,
        U extends VectorBase<?>, V extends VectorBase<?>> {


    /**
     * Solves the linear system of equations given by {@code A*x=b} for {@code x}.
     * @param A Coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     */
    V solve(T A, U b);
}
