/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.core;


/**
 * This interface specifies methods which provide properties of a tensor. All tensors should implement this interface.
 */
interface TensorPropertiesMixin {


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    double min();


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    double max();


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     * @return The minimum value, in absolute value, in this tensor.
     */
    double minAbs();


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     * @return The maximum value, in absolute value, in this tensor.
     */
    double maxAbs();


    /**
     * Finds the indices of the minimum value in this tensor.
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argMin();


    /**
     * Finds the indices of the maximum value in this tensor.
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    int[] argMax();


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     * @return the 2-norm of this tensor.
     */
    double norm();


    /**
     * Computes the p-norm of this tensor.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    double norm(double p);


    /**
     * Computes the maximum/infinite norm of this tensor.
     * @return The maximum/infinite norm of this tensor.
     */
    double infNorm();
}
