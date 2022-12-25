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

package com.flag4j.operations.common.real;

/**
 * This class provides low level methods for computing operations on real tensors. These methods can be applied to
 * either sparse or dense real tensors.
 */
public class RealOperations {

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
}
