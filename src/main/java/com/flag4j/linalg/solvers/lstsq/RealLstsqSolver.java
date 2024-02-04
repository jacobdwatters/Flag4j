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


import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.linalg.decompositions.qr.QRDecompositionOld;
import com.flag4j.linalg.decompositions.qr.RealQRDecompositionOld;
import com.flag4j.linalg.solvers.exact.triangular.RealBackSolver;


/**
 * This class solves a linear system of equations {@code Ax=b} in a least-squares sense. That is,
 * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
 * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
 * This is done using a {@link QRDecompositionOld}.
 */
public class RealLstsqSolver extends LstsqSolver<Matrix, Vector> {


    /**
     * Constructs a least-squares solver to solve a system {@code Ax=b} in a least square sense. That is,
     * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
     * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
     */
    public RealLstsqSolver() {
        super(new RealQRDecompositionOld(), new RealBackSolver());
    }
}
