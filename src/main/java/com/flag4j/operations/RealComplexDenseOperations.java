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
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;


/**
 * This class provides low level methods for computing operations with at least one real tensor
 * and at least one complex tensor.
 */
public final class RealComplexDenseOperations {

    private RealComplexDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static CNumber[] add(CNumber[] src1, Shape shape1, double[] src2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);

        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i].add(src2[i]);
        }

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     */
    public static CNumber[] add(double[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = a.add(src1[i]);
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
    public static CNumber[] sub(CNumber[] src1, Shape shape1, double[] src2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);

        CNumber[] diff = new CNumber[src1.length];

        for(int i=0; i<diff.length; i++) {
            diff[i] = src1[i].sub(src2[i]);
        }

        return diff;
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
    public static CNumber[] sub(double[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);

        CNumber[] diff = new CNumber[src1.length];

        for(int i=0; i<diff.length; i++) {
            diff[i] = new CNumber(src1[i]-src2[i].re, -src2[i].im);
        }

        return diff;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static CNumber[] sub(double[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = new CNumber(src1[i]-a.re, -a.im);
        }

        return sum;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static CNumber[] scalMult(double[] entries, CNumber a) {
        CNumber[] product = new CNumber[entries.length];

        for(int i=0; i<product.length; i++) {
            product[i] = a.mult(entries[i]);
        }

        return product;
    }
}
