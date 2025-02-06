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

package org.flag4j.linalg.solvers;


/**
 * <p>Interface representing a linear system solver for tensor equations. Implementations of this interface
 * provide methods to solve linear equations involving tensors, such as <i>AX=B</i>, where
 * <i>A</i>, <i>B</i>, and <i>X</i> are tensors.
 *
 * <p>Solvers may compute exact solutions or approximate solutions in a least squares sense, depending on the
 * properties of the tensor equation.
 *
 * @param <T> The type of tensor in the linear system.
 */
public interface LinearSolver<T> {

    /**
     * Solves the linear tensor equation <i>AX=B</i> for the tensor <i>X</i>. The multiplication <i>AX</i> is defined such that
     * it performs a tensor dot product over all indices of <i>X</i> with the rightmost indices of <i>A</i>, equivalent to
     * {@code A.tensorDot(X, X.getRank())}.
     *
     * @param A The coefficient tensor in the linear system.
     * @param B The constant tensor in the linear system.
     * @return The solution tensor <i>X</i> satisfying <i>AX=B</i>.
     */
    T solve(T A, T B);
}
