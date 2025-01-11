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

package org.flag4j.linalg;

import org.flag4j.algebraic_structures.Ring;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;


/**
 * A utility class for computing vector norms, including various types of L<sub>p</sub> norms,
 * with support for both dense and sparse vectors. This class provides methods to compute norms
 * for vectors with real entries as well as vectors with entries that belong to a {@link Ring}.
 *
 * <p>The methods in this class utilize scaling internally when computing the L<sub>p</sub> norm to protect against
 * overflow and underflow for very large or very small values of {@code p} (in absolute value).
 *
 * <p><strong>Note:</strong> When {@code p < 1}, the results of the L<sub>p</sub> norm methods are not
 * technically true mathematical norms but may still be useful for numerical tasks. However, {@code p = 0}
 * will result in {@link Double#NaN}.
 *
 * <p>This class is designed to be stateless and is not intended to be instantiated.
 */
public final class VectorNorms {

    private VectorNorms() {
        // Hide default constructor for utility class
    }

    /**
     * <p>Computes the L<sub>2</sub> (Euclidean) norm of a real dense or sparse vector.
     * <p>Zeros do not contribute to this norm so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero data if vector is sparse) to compute norm of.
     * @return L<sub>2</sub> (Euclidean) norm
     */
    public static double norm(double... src) {
        return scaledL2Norm(src);
    }


    /**
     * <p>Computes the L<sub>2</sub> (Euclidean) norm of a dense or sparse vector whose entries are members of a {@link Ring}.
     * <p>Zeros do not contribute to this norm so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero data if vector is sparse) to compute norm of.
     * @return L<sub>2</sub> (Euclidean) norm
     */
    public static <T extends Ring<T>> double norm(T... src) {
        return scaledL2Norm(src);
    }


    /**
     * <p>Computes the L<sub>p</sub> norm (or p-norm) of a real dense or sparse vector.
     * <p>Some common norms:
     * <ul>
     *     <li>{@code p=1}: The taxicab, city block, or Manhattan norm.</li>
     *     <li>{@code p=2}: The Euclidean or L<sub>2</sub> norm.</li>
     * </ul>
     *
     * <p>Zeros do not contribute to this norm so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero data if vector is sparse).
     * @param p The {@code p} value in the {@code p}-norm. When {@code p < 1}, the result of this method is not technically a
     * true mathematical norm. However, it may be useful for various numerical tasks.
     * <ul>
     *     <li>If {@code p} is finite, then the norm is computed as if by:
     *     <pre>{@code
     *     int norm = 0;
     *
     *     for(double v : src)
     *         norm += Math.pow(Math.abs(v), p);
     *
     *     return Math.pow(norm, 1.0/p);
     *     }</pre>
     *     </li>
     *     <li>If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.</li>
     *     <li>If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.</li>
     * </ul>
     *
     * <p>Warning, if {@code p} is very large in absolute value, overflow errors may occur.
     * @return The {@code p}-norm of the vector.
     */
    public static double norm(double[] src, double p) {
        if (src.length == 0) return 0;

        if(p == Double.POSITIVE_INFINITY) {
            return RealProperties.maxAbs(src); // Maximum norm.
        } else if(p == Double.NEGATIVE_INFINITY) {
            return RealProperties.minAbs(src); // Minimum "norm".
        } else if(p == 1) {
            double norm = 0;
            for(double v : src)
                norm += Math.abs(v);

            return norm;
        } else if(p == 2) {
            return scaledL2Norm(src);
        } else {
            return scaledLpNorm(src, p);
        }
    }


    /**
     * <p>Computes the L<sub>p</sub> norm (or p-norm) of a dense or sparse vector whose entries are members of a {@link Ring}.
     * <p>Some common norms:
     * <ul>
     *     <li>{@code p=1}: The taxicab, city block, or Manhattan norm.</li>
     *     <li>{@code p=2}: The Euclidean or L<sub>2</sub> norm.</li>
     * </ul>
     *
     * <p>Zeros do not contribute to this norm so this function may be called on the entries of a dense vector or the non-zero entries
     * of a sparse vector.
     *
     * @param src Entries of the vector (or non-zero data if vector is sparse).
     * @param p The {@code p} value in the {@code p}-norm. When {@code p < 1}, the result of this method is not technically a
     * true mathematical norm. However, it may be useful for various numerical tasks.
     * <ul>
     *     <li>If {@code p} is finite, then the norm is computed as if by:
     *     <pre>{@code
     *     int norm = 0;
     *
     *     for(double v : src)
     *         norm += Math.pow(Math.abs(v), p);
     *
     *     return Math.pow(norm, 1.0/p);
     *     }</pre>
     *     </li>
     *     <li>If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.</li>
     *     <li>If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.</li>
     * </ul>
     *
     * <p>Warning, if {@code p} is very large in absolute value, overflow errors may occur.
     * @return The {@code p}-norm of the vector.
     */
    public static <T extends Ring<T>> double norm(T[] src, double p) {
        if (src.length == 0) return 0;

        if(p == Double.POSITIVE_INFINITY) {
            return CompareRing.maxAbs(src); // Maximum norm.
        } else if(p == Double.NEGATIVE_INFINITY) {
            return CompareRing.minAbs(src); // Minimum "norm".
        } else if(p == 1) {
            double norm = 0;
            for(T v : src)
                norm += v.abs();

            return norm;
        } else if(p == 2) {
            return scaledL2Norm(src);
        } else {
            return scaledLpNorm(src, p);
        }
    }


    /**
     * Computes the scaled L<sub>p</sub> norm of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the L<sub>p</sub> norm for large or small values of {@code p}.
     *
     * @param src The input vector (or non-zero values if vector is sparse) whose L<sub>p</sub> norm is to be computed.
     * @param p The value of {@code p} for the L<sub>p</sub> norm.
     * @return The scaled L<sub>p</sub> norm of the input vector.
     */
    private static double scaledLpNorm(double[] src, double p) {
        // Find the maximum absolute value in the vector.
        double maxAbs = RealProperties.maxAbs(src);

        if (maxAbs == 0.0) return 0.0; // Quick return for zero vector.

        // Compute the p-norm using scaled values.
        double sum = 0;
        for (double v : src)
            sum += Math.pow(Math.abs(v) / maxAbs, p);

        // Ensure result is properly scaled back up.
        return maxAbs * Math.pow(sum, 1.0 / p);
    }


    /**
     * Computes the scaled L<sub>2</sub> norm (Euclidean norm) of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the L<sub>2</sub> norm for vectors with very large or very small values.
     *
     * @param src The input vector (or non-zero entries if the vector is sparse) whose L<sub>2</sub> norm is to be computed.
     * @return The scaled L<sub>2</sub> norm of the input vector.
     */
    private static double scaledL2Norm(double[] src) {
        // Find the maximum absolute value in the vector.
        double maxAbs = RealProperties.maxAbs(src);

        if (maxAbs == 0.0) return 0.0; // Quick return for zero vector.

        // Compute norm as a = |max(src)|, ||src|| = a*||src * (1/a)|| to help protect against overflow.
        double sum = 0;
        for(double v : src) {
            double vScaled = v/maxAbs;
            sum += vScaled*vScaled;
        }

        // Ensure result is properly scaled back up.
        return Math.sqrt(sum)*maxAbs;
    }


    /**
     * Computes the scaled L<sub>p</sub> norm of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the L<sub>p</sub> norm for large or small values of {@code p}.
     *
     * @param src The input vector (or non-zero values if vector is sparse) whose L<sub>p</sub> norm is to be computed.
     * @param p The value of {@code p} for the L<sub>p</sub> norm.
     * @return The scaled L<sub>p</sub> norm of the input vector.
     */
    private static <T extends Ring<T>> double scaledLpNorm(T[] src, double p) {
        // Find the maximum absolute value in the vector.
        double maxAbs = CompareRing.maxAbs(src);

        if (maxAbs == 0.0) return 0.0; // Quick return for zero vector.

        // Compute the p-norm using scaled values.
        double sum = 0;
        for (T v : src)
            sum += Math.pow(v.abs() / maxAbs, p);

        // Ensure result is properly scaled back up.
        return maxAbs * Math.pow(sum, 1.0 / p);
    }


    /**
     * Computes the scaled L<sub>2</sub> norm (Euclidean norm) of a vector.
     * This method uses scaling to protect against numerical instability such as overflow or underflow
     * when computing the L<sub>2</sub> norm for vectors with very large or very small values.
     *
     * @param src The input vector (or non-zero entries if the vector is sparse) whose L<sub>2</sub> norm is to be computed.
     * @return The scaled L<sub>2</sub> norm of the input vector.
     */
    private static <T extends Ring<T>> double scaledL2Norm(T[] src) {
        // Find the maximum absolute value in the vector.
        double maxAbs = CompareRing.maxAbs(src);

        if (maxAbs == 0.0) return 0.0; // Quick return for zero vector.

        // Compute norm as a = |max(src)|, ||src|| = a*||src * (1/a)|| to help protect against overflow.
        double sum = 0;
        for(T v : src) {
            double vScaled = v.mag() / maxAbs;
            sum += vScaled*vScaled;
        }

        // Ensure result is properly scaled back up.
        return Math.sqrt(sum)*maxAbs;
    }
}
