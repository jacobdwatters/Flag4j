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

package com.flag4j.operations.common.real;



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
    public static double norm(double[] src) {
        double norm = 0;

        for(int i=0; i<src.length; i++) {
            norm += src[i]*src[i];
        }

        return Math.sqrt(norm);
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

            for(int i=0; i<src.length; i++) {
                norm += Math.pow(Math.abs(src[i]), p);
            }

            return Math.pow(norm, 1.0/p);
        }
    }
}
