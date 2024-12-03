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

package org.flag4j.linalg;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.util.ErrorMessages;


/**
 * Utility class for computing norms of vectors.
 */
public final class VectorNorms {

    private VectorNorms() {
        // Hide default constructor for utility class
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(Vector, double) norm(src, 2)}.
     *
     * @param src Vector to compute norm of.
     * @return the 2-norm of this vector.
     */
    public static double norm(Vector src) {
        return norm(src.data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param src Vector to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this vector.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(Vector src, double p) {
        return norm(src.data, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(Vector, double) norm(2)}.
     *
     * @param src Vector to compute norm of.
     * @return the 2-norm of this vector.
     */
    public static double norm(CooVector src) {
        return norm(src.data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param src Vector to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this vector.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CooVector src, double p) {
        return norm(src.data, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(CVector, double) norm(2)}.
     *
     * @return the 2-norm of this vector.
     */
    @Deprecated
    public static double norm(CVector src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * Computes the p-norm of this vector. Warning, if p is large in absolute value, overflow issues may occur.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm. <br>
     *          - If p is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.
     * @return The p-norm of this vector.
     */
    @Deprecated
    public static double norm(CVector src, double p) {
        return VectorNorms.norm(src.data, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(CVector, double) norm(2)}.
     *
     * @return the 2-norm of this vector.
     */
    public static <T extends Field<T>> double norm(FieldVector<T> src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * Computes the p-norm of this vector. Warning, if p is large in absolute value, overflow issues may occur.
     *
     * @param p The {@code p} value in the p-norm:
     * <ul>
     *     <li>If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.</li>
     *     <li>If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.</li>
     * </ul>
     *
     * @return The p-norm of this vector.
     */
    public static <T extends Field<T>> double norm(FieldVector<T> src, double p) {
        return VectorNorms.norm(src.data, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(CooCVector, double) norm(src, 2)}.
     *
     * @param src Vector to compute norm of.
     * @return the 2-norm of this vector.
     */
    public static double norm(CooCVector src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param src Vector to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this vector.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CooCVector src, double p) {
        return VectorNorms.norm(src.data, p);
    }


//    /**
//     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
//     * @param src The vector to compute the norm of.
//     * @return The infinity norm of the source vector.
//     */
//    public static double infNorm(CooVector src) {
//        return src.maxAbs();
//    }


//    /**
//     * Computes the infinity norm of a vector. That is, the largest absolute value.
//     * @param src The vector to compute the norm of.
//     * @return The infinity norm of the source vector.
//     */
//    public static double infNorm(CooCVector src) {
//        return src.maxAbs();
//    }
//
//
//    /**
//     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
//     * @param src The vector to compute the norm of.
//     * @return The infinity norm of the source vector.
//     */
//    public static double infNorm(Vector src) {
//        return src.maxAbs();
//    }
//
//
//    /**
//     * Computes the infinity norm of a vector. That is, the largest absolute value.
//     * @param src The vector to compute the norm of.
//     * @return The infinity norm of the source vector.
//     */
//    @Deprecated
//    public static double infNorm(CVector src) {
//        return src.maxAbs();
//    }
//
//
//    /**
//     * Computes the infinity norm of a vector. That is, the largest absolute value.
//     * @param src The vector to compute the norm of.
//     * @return The infinity norm of the source vector.
//     */
//    public static double infNorm(FieldVector src) {
//        return src.maxAbs();
//    }


    // ---------------------------------------------- Low-level Implementations ----------------------------------------------
    /**
     * Computes the 2-norm of a vector.
     * @param src Entries of the vector (or non-zero data if vector is sparse).
     * @return The 2-norm of the vector.
     */
    public static double norm(double... src) {
        double norm = 0;
        double maxAbs = RealProperties.maxAbs(src);
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
     * @param src Entries of the vector (or non-zero data if vector is sparse).
     * @return The 2-norm of the vector.
     */
    public static <T extends Ring<T>> double norm(T... src) {
        double norm = 0;
        double scaledMag;
        double maxAbs = CompareRing.maxAbs(src);

        if(maxAbs == 0) return 0; // Early return for zero norm.

        double maxAbsRecip = 1.0 / maxAbs;

        // Compute norm as a = |max(src)|, ||src|| = a*||src * (1/a)|| to help protect against over/underflow.
        for(Ring<T> value : src) {
            scaledMag = value.mag() * maxAbsRecip;
            norm += scaledMag*scaledMag;
        }

        return Math.sqrt(norm)*maxAbs;
    }


    /**
     * Computes the {@code p}-norm of a vector.
     * @param src Entries of the vector (or non-zero data if vector is sparse).
     * @param p The {@code p} value in the {@code p}-norm:
     * <ul>
     *     <li>If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm.</li>
     *     <li>If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm.</li>
     * </ul>
     *
     * <p>Warning, if {@code p} is very large in absolute value, overflow errors may occur.</p>
     * @return The {@code p}-norm of the vector.
     */
    public static double norm(double[] src, double p) {
        if(Double.isInfinite(p)) {
            if(p > 0) return RealProperties.maxAbs(src); // Maximum norm.
            else return RealProperties.minAbs(src); // Minimum norm.
        } else {
            double norm = 0;

            for(double v : src)
                norm += Math.pow(Math.abs(v), p);

            return Math.pow(norm, 1.0/p);
        }
    }


    /**
     * Computes the {@code p}-norm of a vector.
     * @param src Entries of the vector (or non-zero data if vector is sparse).
     * @param p The {@code p} value in the {@code p}-norm. <br>
     *          - If {@code p} is {@link Double#POSITIVE_INFINITY}, then this method computes the maximum/infinite norm. <br>
     *          - If {@code p} is {@link Double#NEGATIVE_INFINITY}, then this method computes the minimum norm. <br>
     *          Warning, if {@code p} is large in absolute value, overflow errors may occur.
     * @return The {@code p}-norm of the vector.
     */
    public static <T extends Ring<T>> double norm(T[] src, double p) {
        if(Double.isInfinite(p)) {
            if(p > 0) return CompareRing.maxAbs(src); // Maximum norm.
            else return CompareRing.minAbs(src); // Minimum norm.
        } else {
            double norm = 0;

            for(T value : src)
                norm += Math.pow(value.mag(), p);

            return Math.pow(norm, 1.0/p);
        }
    }
}
