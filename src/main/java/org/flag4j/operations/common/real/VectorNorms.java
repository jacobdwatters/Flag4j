/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.operations.common.real;


import org.flag4j.complex_numbers.CNumber;
import org.flag4j.operations.common.complex.AggregateComplex;

/**
 * This class contains low level implementations of vector norms for real valued vectors.
 */
public class VectorNorms {

    private VectorNorms() {
        // Hide default constructor for utility class.
        throw new IllegalStateException();
    }


    /**
     * Computes the 2-norm of a vector.
     * @param src Entries of the vector (or non-zero entries if vector is sparse).
     * @return The 2-norm of the vector.
     */
    public static double norm(double... src) {
        double norm = 0;
        double maxAbs = AggregateReal.maxAbs(src);
        if(maxAbs == 0) return 0; // Early return for zero norm.

        // Compute norm as a = |max(src)|, ||src|| = a*||src * (1/a)|| to help protect against overflow.
        for(double v : src) {
            double vScaled = v/maxAbs;
            norm += vScaled*vScaled;
        }

        return Math.sqrt(norm)*maxAbs;
    }


    /**
     * Computes the 2-norm of a vector.
     * @param src Entries of the vector (or non-zero entries if vector is sparse).
     * @return The 2-norm of the vector.
     */
    public static double norm(CNumber... src) {
        double norm = 0;
        double scaledMag;
        double maxAbs = AggregateComplex.maxAbs(src);

        if(maxAbs == 0) return 0; // Early return for zero norm.

        // Compute norm as a = |max(src)|, ||src|| = a*||src * (1/a)|| to help protect against overflow.
        for(CNumber cNumber : src) {
            scaledMag = cNumber.mag() / maxAbs;
            norm += scaledMag*scaledMag;
        }

        return Math.sqrt(norm)*maxAbs;
    }


    /**
     * Computes the {@code p}-norm of a vector.
     * @param src Entries of the vector (or non-zero entries if vector is sparse).
     * @param p The {@code p} value in the {@code p}-norm. <br>
     *          - If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm. <br>
     *          - If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm. <br>
     *          Warning, if {@code p} is large in absolute value, overflow errors may occur.
     * @return The {@code p}-norm of the vector.
     */
    public static double norm(double[] src, double p) {
        if(Double.isInfinite(p)) {
            if(p > 0) {
                return AggregateReal.maxAbs(src); // Maximum / infinite norm.
            } else {
                return AggregateReal.minAbs(src); // Minimum norm.
            }
        } else {
            double norm = 0;

            for(double v : src) {
                norm += Math.pow(Math.abs(v), p);
            }

            return Math.pow(norm, 1.0/p);
        }
    }


    /**
     * Computes the {@code p}-norm of a vector.
     * @param src Entries of the vector (or non-zero entries if vector is sparse).
     * @param p The {@code p} value in the {@code p}-norm. <br>
     *          - If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm. <br>
     *          - If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm. <br>
     *          Warning, if {@code p} is large in absolute value, overflow errors may occur.
     * @return The {@code p}-norm of the vector.
     */
    public static double norm(CNumber[] src, double p) {
        if(Double.isInfinite(p)) {
            if(p > 0) {
                return AggregateComplex.maxAbs(src); // Maximum / infinite norm.
            } else {
                return AggregateComplex.minAbs(src); // Minimum norm.
            }
        } else {
            double norm = 0;

            for(CNumber cNumber : src) {
                norm += Math.pow(cNumber.mag(), p);
            }

            return Math.pow(norm, 1.0/p);
        }
    }
}
