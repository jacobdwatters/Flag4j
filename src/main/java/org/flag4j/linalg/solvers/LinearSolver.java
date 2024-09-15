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

package org.flag4j.linalg.solvers;


import org.flag4j.arrays.backend.TensorMixin;
import org.flag4j.arrays.backend.TensorOverSemiRing;


/**
 * <p>This interface specifies methods which all linear system solvers should implement.</p>
 *
 * <p>Solvers may solve in an exact sense or in a least squares sense.</p>
 *
 * @param <T> Type of the tensor in the linear system.
 */
public interface LinearSolver<T extends TensorMixin<T, ?>> {

    /**
     * Solves the linear tensor equation given by A*X=B for the tensor X. All indices of X are summed over in
     * the tensor product with the rightmost indices of A as if by
     * {@link org.flag4j.arrays.backend.TensorOverSemiRing#tensorDot(TensorOverSemiRing, int) A.tensorDot(X, X.getRank())}.
     * @param A Coefficient tensor in the linear system.
     * @param B Tensor of constants in the linear system.
     * @return The solution to the tensor X in the linear system A*X=B.
     */
    T solve(T A, T B);
}
