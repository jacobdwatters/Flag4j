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
 * This solver solves linear systems of equations where the coefficient matrix in an upper triangular real dense matrix
 * and the constant vector is a real dense vector.
 */
public class RealBackSolver implements LinearSolver<Matrix, Vector, Vector> {

    // TODO: If a diagonal entry is zero, back-solve fails, add error should be thrown (system is singular).

    /**
     * Solves the linear system of equations given by {@code U*x=b} where the coefficient matrix {@code U}
     * is an upper triangular matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system. If {@code U} is not upper triangular, it
     *          will be treated as such.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     */
    @Override
    public Vector solve(Matrix U, Vector b) {
        double sum;
        int uIndex;
        int n = b.size;
        Vector x = new Vector(U.numRows);

        x.entries[n-1] = b.entries[n-1]/U.entries[n*n-1];

        for(int i=n-2; i>-1; i--) {
            sum = 0;
            uIndex = i*U.numCols;

            for(int j=i+1; j<n; j++) {
                sum += U.entries[uIndex + j]*x.entries[j];
            }

            x.entries[i] = (b.entries[i]-sum)/U.entries[i*(n+1)];
        }

        return x;
    }



    public Vector solveRect(Matrix U, Vector b) {
        Vector x = new Vector(U.numRows);



        return x;
    }
}
