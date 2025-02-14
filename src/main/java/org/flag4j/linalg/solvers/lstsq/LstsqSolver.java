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


import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.linalg.decompositions.unitary.UnitaryDecomposition;
import org.flag4j.linalg.solvers.LinearMatrixSolver;


/**
 * <p>An abstract solver for linear systems of the form <strong>Ax = b</strong> or <strong>AX = B</strong>
 * in a <em>least-squares</em> sense. The system may be under-, well-, or over-determined.
 * Specifically, this solver minimizes the sum of squared residuals:
 * <ul>
 *   <li><strong>||Ax - b||<sub>2</sub><sup>2</sup></strong> for vector-based problems, or</li>
 *   <li><strong>||AX - B||<sub>F</sub><sup>2</sup></strong> for matrix-based where problems</li>
 * </ul>
 * where <strong>||&middot;||<sub>2</sub></strong> is the Euclidean vector norm and <strong>||&middot;||<sub>F</sub></strong>
 * is the Frobenius norm. This is equivalent to solving the normal equations given by:
 * <strong>A<sup>H</sup>Ax = A<sup>H</sup>b</strong> <em>or</em> <strong>A<sup>H</sup>AX = A<sup>H</sup>B</strong>
 *
 * <h2>Usage:</h2>
 * <p>A single system may be solved by calling either {@link #solve(MatrixMixin, VectorMixin)} or
 * {@link #solve(MatrixMixin, VectorMixin)}.
 *
 * <p>Instances of this solver may also be used to efficiently solve many systems of the form <strong>Ax = b</strong> or
 * <strong>AX = B</strong>
 * for the same coefficient matrix <strong>A</strong> but numerous constant vectors/matrices <strong>b</strong> or
 * <strong>B</strong>. To do this, the workflow would be as follows:
 * <ol>
 *     <li>Create a concrete instance of {@code LstsqSolver}.</li>
 *     <li>Call {@link #decompose(MatrixMixin) decompse(A)} once on the coefficient matrix <strong>A</strong>.</li>
 *     <li>Call {@link #solve(VectorMixin) solve(b)} or {@link #solve(MatrixMixin) solve(B)} as many times as needed to solve each
 *     system for with the various <strong>b</strong> vectors and/or <strong>B</strong> matrices. </li>
 * </ol>
 *
 * <b>Note:</b> Any call made to one of the following methods after a call to {@link #decompose(MatrixMixin) decompse(A)} will
 * override the coefficient matrix set that call:
 * <ul>
 *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
 *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <p>Minimizing the sum of squared residuals is achieved by computing a QR decomposition of the coefficient
 * matrix <strong>A</strong>:
 * <blockquote>
 *    <strong>A = QR</strong>
 * </blockquote>
 * where <strong>Q</strong> is a unitary/orthonormal matrix and <strong>R</strong> is an upper triangular matrix.
 * The normal equations then reduces to:
 * <blockquote>
 *   <strong>A<sup>H</sup>Ax = A<sup>H</sup>b</strong> <em>or</em> <strong>A<sup>H</sup>AX = A<sup>H</sup>B</strong> <br>
 *
 *   &Implies; <strong>(QR)<sup>H</sup>QRx = (QR)<sup>H</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>(QR)<sup>H</sup>QRX = (QR)<sup>H</sup>B</strong>  <br>
 *
 *   &Implies; <strong>R<sup>H</sup>Q<sup>H</sup>QRx = R<sup>H</sup>Q<sup>H</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>R<sup>H</sup>Q<sup>H</sup>QRX = R<sup>H</sup>Q<sup>H</sup>B</strong> <br>
 *
 *   &Implies; <strong>R<sup>H</sup>Rx = R<sup>H</sup>Q<sup>H</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>R<sup>H</sup>RX = R<sup>H</sup>Q<sup>H</sup>B</strong>
 *   since <strong>Q</strong> is unitary/orthonormal.<br>
 *
 *   &Implies; <strong>Rx = Q<sup>H</sup>b</strong>
 *   &nbsp; <em>or</em> &nbsp; <strong>RX = Q<sup>H</sup>B</strong>
 * </blockquote>
 * which is easily solved by back-substitution on <strong>R</strong>. In the real case, <strong>Q<sup>H</sup></strong> simply
 * becomes <strong>Q<sup>T</sup></strong>.
 *
 * @param <T> The matrix type in the linear system.
 * @param <U> The vector type in the linear system.
 */
public abstract class LstsqSolver<T extends MatrixMixin<T, ?, U, ?>, U extends VectorMixin<U, T, ?, ?>>
        implements LinearMatrixSolver<T, U> {


    /**
     * Solver for system with an upper triangular coefficient matrix.
     */
    protected final LinearMatrixSolver<T, U> backSolver;
    /**
     * Decomposer to compute the QR decomposition for using the least-squares solver.
     */
    protected final UnitaryDecomposition<T, ?> qr;
    /**
     * The Hermitian transpose of the unitary matrix, <strong>Q</strong>, from the QR decomposition.
     */
    protected T Qh;
    /**
     * The upper triangular matrix, <strong>R</strong>, from the QR decomposition.
     */
    protected T R;

    /**
     * Constructs a least-squares solver with a specified decomposer to use in the QR decomposition.
     * @param qr The QR decomposer to use in the solver.
     * @param backSolver The solver to solve the upper triangular system resulting from the QR decomposition
     *                   which is equivalent to solving the normal equations
     */
    protected LstsqSolver(UnitaryDecomposition<T, ?> qr, LinearMatrixSolver<T, U> backSolver) {
        this.qr = qr;
        this.backSolver = backSolver;
    }


    /**
     * Solves the linear system given by <strong>Ax = b</strong> in a least-squares sense.
     *
     * <p><strong>Note</strong>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <strong>A</strong>, in the linear system.
     * @param b Constant vector, <strong>b</strong>, in the linear system.
     * @return The least-squares solution to <strong>x</strong> in the linear system <strong>Ax = b</strong>.
     */
    @Override
    public U solve(T A, U b) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, (U) Qh.mult(b));
    }


    /**
     * Solves the linear system of equation given by <strong>AX = B</strong> for the matrix <strong>X</strong> in a 
     * least-squares sense.
     *
     * <p><strong>Note</strong>: Any call of this method will override the coefficient matrix specified in any previous calls to
     * {@link #decompose(MatrixMixin)} on the same solver instance.
     *
     * @param A Coefficient matrix, <strong>A</strong>, in the linear system.
     * @param B Constant matrix, <strong>B</strong>, in the linear system.
     * @return The solution to <strong>x</strong> in the linear system <strong>AX = B</strong>.
     */
    @Override
    public T solve(T A, T B) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, (T) Qh.mult(B));
    }


    /**
     * Solves the linear system given by <strong>Ax = b</strong> in a least-squares sense.
     *
     * @param b Constant vector, <strong>b</strong>, in the linear system.
     * @return The solution to <strong>x</strong> in the linear system <strong>Ax = b</strong> for the last <strong>A</strong> passed to
     * {@link #decompose(MatrixMixin)}.
     * @throws IllegalStateException If no coefficient matrix has been specified for this solver by first calling one of the following:
     * <ul>
     *     <li>{@link #decompose(MatrixMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
     * </ul>
     */
    public U solve(U b) {
        if (Qh == null) {
            throw new IllegalStateException("Coefficient matrix has not been specified for this solver." +
                    "\nMust call decompose(...) or a solve(...) which accepts a coefficient matrix first.");
        }

        return backSolver.solve(R, (U) Qh.mult(b));
    }


    /**
     * Solves the set of linear system of equations given by <strong>AX = B</strong> for the matrix <strong>X</strong> in a least-squares sense.
     *
     * @param B Constant matrix, <strong>B</strong>, in the linear system.
     * @return The solution to <strong>X</strong> in the linear system <strong>AX = B</strong> for the last <strong>A</strong> passed to
     * {@link #decompose(MatrixMixin)}.
     * @throws IllegalStateException If no coefficient matrix has been specified for this solver by first calling one of the following:
     * <ul>
     *     <li>{@link #decompose(MatrixMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, VectorMixin)}</li>
     *     <li>{@link #solve(MatrixMixin, MatrixMixin)}</li>
     * </ul>
     */
    public T solve(T B) {
        if (Qh == null) {
            throw new IllegalStateException("Coefficient matrix has not been specified for this solver." +
                    "\nMust call decompose(...) or a solve(...) which accepts a coefficient matrix first.");
        }

        return backSolver.solve(R, (T) Qh.mult(B));
    }


    /**
     * <p>Computes (or updates) the QR decomposition of the given matrix <strong>A</strong>
     * for use in subsequent solves.
     *
     * <p>Subclasses may override this method to customize how the QR decomposition is computed or updated.
     *
     * @param A Coefficient matrix to decompose.
     */
    public void decompose(T A) {
        qr.decompose(A);
        Qh = qr.getQ().H();
        R = qr.getUpper();
    }
}
