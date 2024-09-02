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

import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.sparse.CooCTensorOld;
import org.flag4j.arrays_old.sparse.CooTensorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core_old.dense_base.ComplexDenseTensorBase;
import org.flag4j.core_old.dense_base.RealDenseTensorBase;
import org.flag4j.operations_old.common.complex.AggregateComplex;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;


/**
 * Utility class for computing "norms" of tensors.
 */
public class TensorNorms {

    private TensorNorms() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
     * @param src The tensor, matrix, or vector to compute the norm of.
     * @return The infinity norm of the source tensor, matrix, or vector.
     */
    public static double infNorm(RealDenseTensorBase<?, ?> src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest value by magnitude.
     * @param src The tensor, matrix, or vector to compute the norm of.
     * @return The infinity norm of the source tensor, matrix, or vector.
     */
    public static double infNorm(ComplexDenseTensorBase<?, ?> src) {
        return src.maxAbs();
    }


    /**
     * Computes the 2-norm of this tensor as if the tensor was a vector (i.e. as if by {@code VectorNorm(TensorOld.toVector())}).
     * This is equivalent to {@link #norm(TensorOld, double) norm(src, 2)}.
     *
     * @param src TensorOld to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(TensorOld src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src TensorOld to compute norm of.
     * @param p The {@code p} value in the p-norm. <br>
     *          - If {@code p} is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(TensorOld src, double p) {
        return tensorNormLp(src.entries, p);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CTensorOld, double) norm(src, 2)}.
     *
     * @return the 2-norm of this tensor.
     */
    public double norm(CTensorOld src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src TensorOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CTensorOld src, double p) {
        return tensorNormLp(src.entries, p);
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @param src TensorOld to compute norm of.
     * @return The maximum/infinite norm of this tensor.
     */
    public double infNorm(CTensorOld src) {
        return AggregateComplex.maxAbs(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooTensorOld, double) norm(src, 2)}.
     *
     * @param src TensorOld to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CooTensorOld src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src TensorOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CooTensorOld src, double p) {
        return tensorNormLp(src.entries, p);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooTensorOld, double) norm(src, 2)}.
     *
     * @param src TensorOld to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CooCTensorOld src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src TensorOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CooCTensorOld src, double p) {
        return tensorNormLp(src.entries, p);
    }


    // -------------------------------------------------- Low-level implementations --------------------------------------------------

    /**
     * Computes the L<sub>2</sub> norm of a tensor.
     * @param src Entries of the tensor.
     * @return The L<sub>2</sub> norm of the tensor.
     */
    protected static double tensorNormL2(double[] src) {
        double norm = 0;

        for(double value : src) {
            norm += Math.pow(Math.abs(value), 2);
        }

        return Math.sqrt(norm);
    }


    /**
     * Computes the L<sub>p</sub> norm of a tensor.
     * @param src Entries of the tensor.
     * @param p The {@code p} parameter of the L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the tensor.
     */
    protected static double tensorNormLp(double[] src, double p) {
        ParameterChecks.ensureNotEquals(0, p);
        double norm = 0;

        for(double value : src) {
            norm += Math.pow(Math.abs(value), p);
        }

        return Math.pow(norm, 1.0/p);
    }


    /**
     * Computes the L<sub>2</sub> norm of a tensor (i.e. the Frobenius norm).
     * @param src Entries of the tensor.
     * @return The L<sub>2</sub> norm of the tensor.
     */
    public static double tensorNormL2(CNumber[] src) {
        double norm = 0;

        for(CNumber cNumber : src) {
            norm += CNumber.pow(cNumber, 2).mag();
        }

        return Math.sqrt(norm);
    }


    /**
     * Computes the L<sub>p</sub> norm of a tensor (i.e. the Frobenius norm).
     * @param src Entries of the tensor.
     * @param p The {@code p} parameter of the L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the tensor.
     */
    public static double tensorNormLp(CNumber[] src, double p) {
        double norm = 0;

        for(CNumber cNumber : src) {
            norm += CNumber.pow(cNumber, p).mag();
        }

        return Math.pow(norm, 1.0/p);
    }
}
