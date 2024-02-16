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

package com.flag4j.linalg.solvers;

import com.flag4j.core.TensorBase;

/**
 * This interface specifies methods which all linear tensor system solvers should implement. Solvers
 * may solve in an exact sense or in a least squares sense.
 * @param <T> Type of the tensors in the linear system.
 */
public interface LinearTensorSolver<T extends TensorBase<T, ?, ?, ?, ?, ?, ?>> {


    /**
     * Solves the linear tensor equation given by {@code A*X=B} for the tensor {@code X}. All indices of {@code X} are summed over in
     * the tensor product with the rightmost indices of {@code A} as if by
     * {@link com.flag4j.core.TensorExclusiveMixin#tensorDot(TensorBase, int)  A.tensorDot(X, X.getRank())}.
     * @param A Coefficient tensor in the linear system.
     * @param B Tensor of constants in the linear system.
     * @return The solution to {@code x} in the linear system {@code A*X=B}.
     */
    T solve(T A, T B);
}
