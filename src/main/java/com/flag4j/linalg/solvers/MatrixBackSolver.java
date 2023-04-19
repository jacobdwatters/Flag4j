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
public class MatrixBackSolver implements LinearSolver<Matrix, Vector, Vector> {

    /**
     * Solves the linear system of equations given by {@code U*x=b} where the coefficient matrix {@code U}
     * is an upper triangular matrix.
     *
     * @param U Upper triangular coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*x=b}.
     * @throws IllegalArgumentException If {@code U} is not upper triangular.
     */
    @Override
    public Vector solve(Matrix U, Vector b) {
        if(!U.isTriU()) {
            throw new IllegalArgumentException("Coefficient matrix must be upper triangular.");
        }

        double sum;
        int uIndex;
        Vector x = new Vector(U.numRows);

        x.entries[U.numRows-1] = b.entries[U.numRows-1]/U.entries[(U.numRows-1)*(U.numCols+1)];

        for(int i=U.numRows-1; i>-1; i--) {
            sum = 0;
            uIndex = i*U.numCols;

            for(int j=i+1; j<U.numRows; j++) {
                sum += U.entries[uIndex + j]*x.entries[j];
            }

            x.entries[i] = (b.entries[i]-sum)/U.entries[i*(U.numCols+1)];
        }

        return x;
    }
}
