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

package com.flag4j.operations.common.complex;


import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.concurrency.util.ErrorMessages;

/**
 * This class provides low level methods for computing operations on real tensors. These methods can be applied to
 * either sparse or dense real tensors.
 */
public class ComplexOperations {

    private ComplexOperations() {
        // Hide constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static CNumber[] sqrt(CNumber[] src) {
        CNumber[] roots = new CNumber[src.length];

        for(int i=0; i<roots.length; i++) {
            roots[i] = CNumber.sqrt(src[i]);
        }

        return roots;
    }


    /**
     * Computes the element-wise absolute value of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise absolute value of the tensor.
     */
    public static CNumber[] abs(CNumber[] src) {
        CNumber[] abs = new CNumber[src.length];

        for(int i=0; i<abs.length; i++) {
            abs[i] = src[i].mag();
        }

        return abs;
    }


    /**
     * Rounds the values of a tensor to the nearest integer. Also see {@link #round(CNumber[], int)}.
     * @param src Entries of the tensor to round.
     * @return The result of rounding all entries of the source tensor to the nearest integer.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static CNumber[] round(CNumber[] src) {
        CNumber[] dest = new CNumber[src.length];

        for(int i=0; i<dest.length; i++) {
            dest[i] = CNumber.round(src[i]);
        }

        return dest;
    }


    /**
     * Rounds the values of a tensor with specified precision. Note, if precision is zero, {@link #round(CNumber[])} is
     * preferred.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e. the number of decimal places).
     * @return The result of rounding all entries of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static CNumber[] round(CNumber[] src, int precision) {
        if(precision<0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(precision));
        }

        CNumber[] dest = new CNumber[src.length];

        for(int i=0; i<dest.length; i++) {
            dest[i] = CNumber.round(src[i], precision);
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
    public static CNumber[] roundToZero(CNumber[] src, double threshold) {
        if(threshold<0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(threshold));
        }

        CNumber[] dest = new CNumber[src.length];

        for(int i=0; i<dest.length; i++) {
            dest[i] = CNumber.roundToZero(src[i], threshold);
        }

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Entries of the tensor.
     * @param factor Scalar value.
     * @return The result of the scalar multiplication of a tensor.
     */
    public static CNumber[] scalMult(CNumber[] src, double factor) {
        CNumber[] dest = new CNumber[src.length];

        for(int i=0; i<src.length; i++) {
            dest[i] = src[i].mult(factor);
        }

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Entries of the tensor.
     * @param factor Scalar value.
     * @return The result of the scalar multiplication of a tensor.
     */
    public static CNumber[] scalMult(CNumber[] src, CNumber factor) {
        CNumber[] dest = new CNumber[src.length];

        for(int i=0; i<src.length; i++) {
            dest[i] = src[i].mult(factor);
        }

        return dest;
    }
}