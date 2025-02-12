/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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
 * <p>Instances of this class solve real linear systems of the form <strong>Ax = b</strong> or <strong>AX = B</strong>
 * in a <em>least-squares</em> sense. The system may be under-, well-, or over-determined.
 * Specifically, this solver minimizes the sum of squared residuals:
 * <ul>
 *   <li><strong>||Ax - b||<sub>2</sub><sup>2</sup></strong> for vector-based problems, or</li>
 *   <li><strong>||AX - B||<sub>F</sub><sup>2</sup></strong> for matrix-based where problems</li>
 * </ul>
 * where <strong>||&middot;||<sub>2</sub></strong> is the Euclidean vector norm and <strong>||&middot;||<sub>F</sub></strong>
 * is the Frobenius norm. This is equivalent to solving the normal equations given by:
 * <strong>A<sup>T</sup>Ax = A<sup>T</sup>b</strong> <em>or</em> <strong>A<sup>T</sup>AX = A<sup>T</sup>B</strong>
 *
 * <h2>Usage:</h2>
 * <p>A single system may be solved by calling either {@link #solve(Matrix, Vector)} or
 * {@link #solve(Matrix, Vector)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form <strong>Ax = b</strong> or
 * <strong>AX = B</strong>
 * for the same coefficient matrix <strong>A</strong> but numerous constant vectors/matrices <strong>b</strong> or
 * <strong>B</strong>. To do this, the workflow would be as follows:
 * <ol>
 *     <li>Create an instance of {@code RealLstsqSolver}.</li>
 *     <li>Call {@link #decompose(Matrix) decompse(A)} once on the coefficient matrix <strong>A</strong>.</li>
 *     <li>Call {@link #solve(Vector) solve(b)} or {@link #solve(Matrix) solve(B)} as many times as needed to solve each
 *     system for with the various <strong>b</strong> vectors and/or <strong>B</strong> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link #decompose(Matrix) decompse(A)} will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link #solve(Matrix, Vector)}</li>
 *     <li>{@link #solve(Matrix, Matrix)}</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <p>Minimizing the sum of squared residuals is achieved by computing a QR decomposition of the coefficient
 * matrix <strong>A</strong>:
 * <blockquote>
 *    <strong>A = QR</strong>
 * </blockquote>
 * where <strong>Q</strong> is an orthonormal matrix and <strong>R</strong> is an upper triangular matrix.
 * The normal equations then reduces to:
 * <blockquote>
 *   <strong>A<sup>T</sup>Ax = A<sup>T</sup>b</strong> <em>or</em> <strong>A<sup>T</sup>AX = A<sup>T</sup>B</strong> <br>
 *
 *   &Implies; <strong>(QR)<sup>T</sup>QRx = (QR)<sup>T</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>(QR)<sup>T</sup>QRX = (QR)<sup>T</sup>B</strong>  <br>
 *
 *   &Implies; <strong>R<sup>T</sup>Q<sup>T</sup>QRx = R<sup>T</sup>Q<sup>T</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>R<sup>T</sup>Q<sup>T</sup>QRX = R<sup>T</sup>Q<sup>T</sup>B</strong> <br>
 *
 *   &Implies; <strong>R<sup>T</sup>Rx = R<sup>T</sup>Q<sup>T</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>R<sup>T</sup>RX = R<sup>T</sup>Q<sup>T</sup>B</strong>
 *   since <strong>Q</strong> is orthonormal.<br>
 *
 *   &Implies; <strong>Rx = Q<sup>T</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>RX = Q<sup>T</sup>B</strong>
 * </blockquote>
 * which is easily solved by back-substitution on <strong>R</strong>. In the real case, <strong>Q<sup>T</sup></strong> simply
 * becomes <strong>Q<sup>T</sup></strong>.
 */
public class RealLstsqSolver extends LstsqSolver<Matrix, Vector> {

    
    /**
     * Creates a solver for solving real linear systems of the form <strong>Ax = b</strong> or <strong>AX = B</strong>
     * in a <em>least-squares</em> sense. The system may be under-, well-, or over-determined.
     * The solution is computed by minimizing the sum of squared residuals:
     *
     * <ul>
     *   <li><strong>||Ax - b||<sub>2</sub><sup>2</sup></strong> for vector-based problems, or</li>
     *   <li><strong>||AX - B||<sub>F</sub><sup>2</sup></strong> for matrix-based where problems</li>
     * </ul>
     */
    public RealLstsqSolver() {
        super(new RealQR(), new RealBackSolver(false));
    }
}
