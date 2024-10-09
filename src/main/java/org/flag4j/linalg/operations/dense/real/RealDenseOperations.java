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

package org.flag4j.linalg.operations.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.linalg.operations.common.real.RealOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * This class provides low level methods for computing operations on real dense tensors.
 */
public final class RealDenseOperations {

    private RealDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param src1 Entries of first Tensor of the addition.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second Tensor of the addition.
     * @param shape2 Shape of second tensor.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays_old are not the same size.
     */
    public static double[] add(final double[] src1, final Shape shape1,
                               final double[] src2, final Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        int length = src1.length;
        double[] sum = new double[length];

        for(int i=0; i<length; i++)
            sum[i] = src1[i] + src2[i];

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws IllegalArgumentException If entry arrays_old are not the same size.
     */
    public static double[] sub(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        double[] sum = new double[src1.length];
        int length = sum.length;

        for(int i=0; i<length; i++)
            sum[i] = src1[i] - src2[i];

        return sum;
    }


    /**
     * Subtracts a scalar from every element of a tensor.
     * @param src Entries of tensor to add scalar to.
     * @param b Scalar to subtract from tensor.
     * @return The tensor scalar subtraction.
     */
    public static double[] sub(double[] src, double b) {
        double[] sum = new double[src.length];
        int length = sum.length;

        for(int i=0; i<length; i++)
            sum[i] = src[i] - b;

        return sum;
    }


    /**
     * Computes element-wise subtraction between tensors and stores the result in the first tensor.
     * @param src1 First tensor in subtraction. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the subtraction.
     * @param shape2 Shape of the second tensor.
     * @throws IllegalArgumentException If tensors are not the same shape.
     */
    public static void subEq(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, length = src1.length; i<length; i++)
            src1[i] -= src2[i];
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static void subEq(double[] src, double b) {
        for(int i=0, length=src.length; i<length; i++)
            src[i] -= b;
    }


    /**
     * Computes element-wise addition between tensors and stores the result in the first tensor.
     * @param src1 First tensor in addition. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the addition.
     * @param shape2 Shape of the second tensor.
     * @throws IllegalArgumentException If tensors are not the same shape.
     */
    public static void addEq(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, length = src1.length; i<length; i++)
            src1[i] += src2[i];
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static void addEq(double[] src, double b) {
        for(int i=0, length = src.length; i<length; i++)
            src[i] += b;
    }


    /**
     * Multiplies all entries in a tensor.
     * @param src The entries of the tensor.
     * @return The product of all entries in the tensor.
     */
    public static double prod(double[] src) {
        if(src == null || src.length == 0) return 0;
        double product=1;

        for(double value : src)
            product *= value;

        return product;
    }


    /**
     * Multiplies all entries in a tensor.
     * @param src The entries of the tensor.
     * @return The product of all entries in the tensor.
     */
    public static int prod(int[] src) {
        if(src == null || src.length == 0) return 0;
        int product=1;

        for(int value : src)
            product *= value;

        return product;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param src Entries of the tensor.
     * @param divisor Scalar to divide by.
     * @return The scalar division of the tensor.
     */
    public static double[] scalDiv(double[] src, double divisor) {
        return RealOperations.scalDiv(src, divisor);
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise reciprocals of the tensor.
     */
    public static double[] recip(double[] src) {
        double[] receps = new double[src.length];

        for(int i=0; i<receps.length; i++)
            receps[i] = 1/src[i];

        return receps;
    }


    /**
     * Adds a scalar to every element of a tensor.
     * @param src src of tensor to add scalar to.
     * @param b Scalar to add to tensor.
     * @return The tensor scalar addition.
     */
    public static double[] add(double[] src, double b) {
        int length = src.length;
        double[] sum = new double[length];

        for(int i=0; i<length; i++)
            sum[i] = src[i] + b;

        return sum;
    }
}
