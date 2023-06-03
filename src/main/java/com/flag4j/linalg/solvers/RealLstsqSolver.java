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
import com.flag4j.linalg.decompositions.QRDecomposition;
import com.flag4j.linalg.decompositions.RealQRDecomposition;


/**
 * This class solves a linear system of equations {@code Ax=b} in a least-squares sense. That is,
 * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
 * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
 * This is done using a {@link QRDecomposition}.
 */
public class RealLstsqSolver extends LstsqSolver<Matrix, Vector, Vector> {

    /**
     * Backwards solver for solving the system of equations formed from the {@code QR} decomposition,
     * {@code Rx=Q<sup>T</sup>b} which is an equivalent system to {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
     */
    private final RealBackSolver backSolver;

    /**
     * Constructs a least-squares solver to solve a system {@code Ax=b} in a least square sense. That is,
     * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
     * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
     */
    protected RealLstsqSolver() {
        super(new RealQRDecomposition());
        backSolver = new RealBackSolver();
    }


    /**
     * Solves the linear system given by {@code Ax=b} in the least-squares sense.
     *
     * @param A Coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The least squares solution to {@code x} in the linear system {@code Ax=b}.
     */
    @Override
    public Vector solve(Matrix A, Vector b) {
        decompose(A); // Compute the QR decomposition of A.
        return backSolver.solve(R, Q.T().mult(b).toVector());
    }
}
