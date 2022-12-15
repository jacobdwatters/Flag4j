/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j.operations;

import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;

/**
 * This class provides low level methods for computing operations on real dense tensors.
 */
public final class RealDenseOperations {

    private RealDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param src1 Entries of first Tensor of the addition.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second Tensor of the addition.
     * @param shape2 Shape of second tensor.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static double[] add(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);
        double[] sum = new double[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i] + src2[i];
        }

        return sum;
    }


    /**
     * Adds a scalar to every element of a tensor.
     * @param src src of tensor to add scalar to.
     * @param b Scalar to add to tensor.
     * @return The tensor scalar addition.
     */
    public static double[] add(double[] src, double b) {
        double[] sum = new double[src.length];

        for(int i=0; i<src.length; i++) {
            sum[i] = src[i] + b;
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static double[] sub(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);
        double[] sum = new double[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i] - src2[i];
        }

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

        for(int i=0; i<src.length; i++) {
            sum[i] = src[i] - b;
        }

        return sum;
    }


    /**
     * Sums all entries in a tensor.
     * @param src Entries of tensor so sum.
     * @return The sum of all src in the tensor.
     */
    public static double sum(double[] src) {
        double sum = 0;

        for(double value : src) {
            sum += value;
        }

        return sum;
    }


    /**
     * Multiplies all entries in a tensor.
     * @param src The entries of the tensor.
     * @return The product of all entries in the tensor.
     */
    public static double prod(double[] src) {
        double product;

        if(src.length > 0) {
            product=1;
            for(double value : src) {
                product *= value;
            }
        } else {
            product=0;
        }

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param src Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static double[] scalMult(double[] src, double a) {
        double[] product = new double[src.length];

        for(int i=0; i<product.length; i++) {
            product[i] = src[i]*a;
        }

        return product;
    }



}
