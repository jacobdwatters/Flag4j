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
import com.flag4j.Vector;


/**
 * This solver solves linear systems of equations where the coefficient matrix in a lower triangular real dense matrix
 * and the constant vector is a real dense vector.
 */
public class RealForwardSolver implements LinearSolver<Matrix, Vector, Vector> {

    /**
     * Performs forward substitution for a unit lower triangular matrix {@code L} and a vector {@code b}.
     * That is, solves the linear system {@code L*x=b} where {@code L} is a lower triangular matrix.
     * @param L Unit lower triangular coefficient matrix. If {@code L} is not unit lower triangular, it will be treated
     *          as such.
     * @param b Constant vector.
     * @return The result of solving the linear system {@code L*x=b} where {@code L} is a lower triangular matrix.
     * @throws IllegalArgumentException If {@code L} is not lower triangular.
     */
    @Override
    public Vector solve(Matrix L, Vector b) {
        double sum;
        int lIndexStart;
        Vector x = new Vector(L.numRows);

        x.entries[0] = b.entries[0];

        for(int i=1; i<L.numRows; i++) {
            sum = 0;
            lIndexStart = i*L.numCols;

            for(int j=i-1; j>-1; j--) {
                sum += L.entries[lIndexStart + j]*x.entries[j];
            }

            x.entries[i] = b.entries[i]-sum;
        }

        return x;
    }
}
