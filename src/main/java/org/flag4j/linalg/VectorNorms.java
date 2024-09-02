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
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.operations_old.common.complex.AggregateComplex;
import org.flag4j.operations_old.common.real.AggregateReal;
import org.flag4j.util.ErrorMessages;


/**
 * Utility class for computing norms of vectors.
 */
public final class VectorNorms {

    private VectorNorms() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(Vector, double) norm(src, 2)}.
     *
     * @param src VectorOld to compute norm of.
     * @return the 2-norm of this vector.
     */
    public static double norm(Vector src) {
        return norm(src.entries);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param src VectorOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this vector.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(Vector src, double p) {
        return norm(src.entries, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(Vector, double) norm(2)}.
     *
     * @param src VectorOld to compute norm of.
     * @return the 2-norm of this vector.
     */
    public static double norm(CooVectorOld src) {
        return norm(src.entries);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param src VectorOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this vector.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CooVectorOld src, double p) {
        return norm(src.entries, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(CVectorOld, double) norm(2)}.
     *
     * @return the 2-norm of this vector.
     */
    @Deprecated
    public static double norm(CVectorOld src) {
        return VectorNorms.norm(src.entries);
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
    public static double norm(CVectorOld src, double p) {
        return VectorNorms.norm(src.entries, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(CVectorOld, double) norm(2)}.
     *
     * @return the 2-norm of this vector.
     */
    public static <T extends Field<T>> double norm(FieldVector<T> src) {
        return VectorNorms.norm(src.entries);
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
        return VectorNorms.norm(src.entries, p);
    }


    /**
     * Computes the 2-norm of this vector. This is equivalent to {@link #norm(CooCVectorOld, double) norm(src, 2)}.
     *
     * @param src VectorOld to compute norm of.
     * @return the 2-norm of this vector.
     */
    public static double norm(CooCVectorOld src) {
        return VectorNorms.norm(src.entries);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param src VectorOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this vector.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CooCVectorOld src, double p) {
        return VectorNorms.norm(src.entries, p);
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
     * @param src The vector to compute the norm of.
     * @return The infinity norm of the source vector.
     */
    public static double infNorm(CooVectorOld src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a vector. That is, the largest absolute value.
     * @param src The vector to compute the norm of.
     * @return The infinity norm of the source vector.
     */
    public static double infNorm(CooCVectorOld src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
     * @param src The vector to compute the norm of.
     * @return The infinity norm of the source vector.
     */
    public static double infNorm(Vector src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a vector. That is, the largest absolute value.
     * @param src The vector to compute the norm of.
     * @return The infinity norm of the source vector.
     */
    @Deprecated
    public static double infNorm(CVectorOld src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a vector. That is, the largest absolute value.
     * @param src The vector to compute the norm of.
     * @return The infinity norm of the source vector.
     */
    public static double infNorm(FieldVector src) {
        return src.maxAbs();
    }


    // ---------------------------------------------- Low-level Implementations ----------------------------------------------
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
    @Deprecated
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
     * Computes the 2-norm of a vector.
     * @param src Entries of the vector (or non-zero entries if vector is sparse).
     * @return The 2-norm of the vector.
     */
    public static double norm(Field... src) {
        double norm = 0;
        double scaledMag;
        double maxAbs = CompareField.maxAbs(src);

        if(maxAbs == 0) return 0; // Early return for zero norm.

        // Compute norm as a = |max(src)|, ||src|| = a*||src * (1/a)|| to help protect against overflow.
        for(Field value : src) {
            scaledMag = value.mag() / maxAbs;
            norm += scaledMag*scaledMag;
        }

        return Math.sqrt(norm)*maxAbs;
    }


    /**
     * Computes the {@code p}-norm of a vector.
     * @param src Entries of the vector (or non-zero entries if vector is sparse).
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
            if(p > 0) return AggregateReal.maxAbs(src); // Maximum norm.
            else return AggregateReal.minAbs(src); // Minimum norm.
        } else {
            double norm = 0;

            for(double v : src)
                norm += Math.pow(Math.abs(v), p);

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
    @Deprecated
    public static double norm(CNumber[] src, double p) {
        if(Double.isInfinite(p)) {
            if(p > 0) return AggregateComplex.maxAbs(src); // Maximum norm.
            else return AggregateComplex.minAbs(src); // Minimum norm.
        } else {
            double norm = 0;

            for(CNumber cNumber : src)
                norm += Math.pow(cNumber.mag(), p);

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
    public static double norm(Field[] src, double p) {
        if(Double.isInfinite(p)) {
            if(p > 0) return CompareField.maxAbs(src); // Maximum norm.
            else return CompareField.minAbs(src); // Minimum norm.
        } else {
            double norm = 0;

            for(Field value : src)
                norm += Math.pow(value.mag(), p);

            return Math.pow(norm, 1.0/p);
        }
    }
}
