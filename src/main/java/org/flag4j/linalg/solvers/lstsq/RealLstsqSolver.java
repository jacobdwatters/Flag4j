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

package org.flag4j.linalg.solvers.lstsq;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.qr.RealQR;
import org.flag4j.linalg.solvers.exact.triangular.RealBackSolver;


/**
 * <p>This class solves a linear system of equations Ax=b in a least-squares sense. That is,
 * minimizes ||Ax-b||<sub>2</sub> which is equivalent to solving the normal equations
 * A<sup>T</sup>Ax=A<sup>T</sup>b.</p>
 *
 * <p>This is done efficiently using a {@link RealQR QR decomposition}.</p>
 */
public class RealLstsqSolver extends LstsqSolver<Matrix, Vector> {

    /**
     * <p>Constructs a least-squares solver to solve a system Ax=b in a least square sense.</p>
     *
     * <p>That is, minimizes ||Ax-b||<sub>2</sub> which is equivalent to solving the normal equations
     * A<sup>T</sup>Ax=A<sup>T</sup>b.</p>
     */
    public RealLstsqSolver() {
        super(new RealQR(), new RealBackSolver());
    }
}
