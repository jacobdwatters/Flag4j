/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

import com.flag4j.Shape;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.math.BigDecimal;
import java.math.RoundingMode;

import static com.flag4j.operations.common.real.Aggregate.maxAbs;

/**
 * This class provides low level methods for computing operations on real tensors. These methods can be applied to
 * either sparse or dense real tensors.
 */
public class RealOperations {

    private RealOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }

    /**
     * Computes the scalar multiplication of a tensor.
     * @param src Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static double[] scalMult(double[] src, double factor) {
        double[] product = new double[src.length];

        for(int i=0; i<product.length; i++) {
            product[i] = src[i]*factor;
        }

        return product;
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static double[] sqrt(double[] src) {
        double[] roots = new double[src.length];

        for(int i=0; i<roots.length; i++) {
            roots[i] = Math.sqrt(src[i]);
        }

        return roots;
    }


    /**
     * Computes the element-wise absolute value of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise absolute value of the tensor.
     */
    public static double[] abs(double[] src) {
        double[] abs = new double[src.length];

        for(int i=0; i<abs.length; i++) {
            abs[i] = Math.abs(src[i]);
        }

        return abs;
    }


    /**
     * Compute the L<sub>p, q</sub> norm of a matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p First parameter in L<sub>p, q</sub> norm.
     * @param q Second parameter in L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} or {@code q} is less than 1.
     */
    public static double matrixNorm(double[] src, Shape shape, double p, double q) {
        ParameterChecks.assertGreaterEq(1, p, q);

        double norm = 0;
        double colSum;
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];

        // TODO: Is transposing first faster here?
        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(Math.abs(src[i*cols + j]), p);
            }
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1/q);
    }


    /**
     * Compute the L<sub>p</sub> norm of a matrix. This is equivalent to passing {@code q=1} to
     * {@link #matrixNorm(double[], Shape, double, double)}
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p Parameter in L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} is less than 1.
     */
    public static double matrixNorm(double[] src, Shape shape, double p) {
        ParameterChecks.assertGreaterEq(1, p);

        double norm = 0;
        double colSum;
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];

        // TODO: Is transposing first faster here?
        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(Math.abs(src[i*cols + j]), p);
            }

            norm += Math.pow(colSum, 1.0/p);
        }

        return norm;
    }


    /**
     * Compute the L<sub>2</sub> norm of a matrix. This is equivalent to passing {@code q=1} to
     * {@link #matrixNorm(double[], Shape, double, double)}
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return The L<sub>2</sub> norm of the matrix.
     */
    public static double matrixNorm(double[] src, Shape shape) {
        double norm = 0;
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];

        double colSum;

        // TODO: Is transposing first faster here?
        for(int j=0; j<cols; j++) {
            colSum = 0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(src[i*cols + j], 2);
            }
            norm += Math.sqrt(colSum);
        }

        return norm;
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    public static double matrixMaxNorm(double[] src) {
        return maxAbs(src);
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    public static double matrixInfNorm(double[] src, Shape shape) {
        int rows = shape.dims[Axis2D.row()];
        int cols = shape.dims[Axis2D.col()];
        double[] rowSums = new double[rows];

        // TODO: Is transposing first faster?
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rowSums[i] += Math.abs(src[i*cols + j]);
            }
        }

        return maxAbs(rowSums);
    }


    /**
     * Rounds the values of a tensor to the nearest integer. Also see {@link #round(double[], int)}.
     * @param src Entries of the tensor to round.
     * @return The result of rounding all entries of the source tensor to the nearest integer.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static double[] round(double[] src) {
        double[] dest = new double[src.length];

        for(int i=0; i<dest.length; i++) {
            dest[i] = Math.round(src[i]);
        }

        return dest;
    }


    /**
     * Rounds the values of a tensor with specified precision. Note, if precision is zero, {@link #round(double[])} is
     * preferred.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e. the number of decimal places).
     * @return The result of rounding all entries of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static double[] round(double[] src, int precision) {
        if(precision<0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(precision));
        }

        BigDecimal bd;
        double[] dest = new double[src.length];

        for(int i=0; i<dest.length; i++) {
            bd = new BigDecimal(Double.toString(src[i]));
            bd = bd.setScale(precision, RoundingMode.HALF_UP);
            dest[i] = bd.doubleValue();
        }

        return dest;
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this tensor is less than
     *                  the threshold in absolute value then it will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If {@code threshold} is negative.
     */
    public static double[] roundToZero(double[] src, double threshold) {
        if(threshold<0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(threshold));
        }

        double[] dest = new double[src.length];

        for(int i=0; i<dest.length; i++) {
            if(Math.abs(src[i]) < threshold) {
                dest[i] = 0;
            } else {
                dest[i] = src[i];
            }
        }

        return dest;
    }
}
