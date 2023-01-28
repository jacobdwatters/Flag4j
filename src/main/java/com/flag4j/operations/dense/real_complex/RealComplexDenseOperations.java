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

package com.flag4j.operations.dense.real_complex;

import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;


/**
 * This class provides low level methods for computing operations with at least one real tensor
 * and at least one complex tensor.
 */
public final class RealComplexDenseOperations {

    private RealComplexDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
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
        ParameterChecks.assertEqualShape(shape1, shape2);

        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = src1[i].add(src2[i]);
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
        ParameterChecks.assertEqualShape(shape1, shape2);

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
        ParameterChecks.assertEqualShape(shape1, shape2);

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
     * Computes element-wise subtraction between tensors and stores the result in the first tensor.
     * @param src1 First tensor in subtraction. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the subtraction.
     * @param shape2 Shape of the second tensor.
     * @throws IllegalArgumentException If tensors are not the same shape.
     */
    public static void subEq(CNumber[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);

        for(int i=0; i<src1.length; i++) {
            src1[i].subEq(src2[i]);
        }
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static void subEq(CNumber[] src, double b) {
        for(int i=0; i<src.length; i++) {
            src[i].subEq(b);
        }
    }


    /**
     * Computes element-wise addition between tensors and stores the result in the first tensor.
     * @param src1 First tensor in addition. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the addition.
     * @param shape2 Shape of the second tensor.
     * @throws IllegalArgumentException If tensors are not the same shape.
     */
    public static void addEq(CNumber[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);

        for(int i=0; i<src1.length; i++) {
            src1[i].addEq(src2[i]);
        }
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static void addEq(CNumber[] src, double b) {
        for(int i=0; i<src.length; i++) {
            src[i].addEq(b);
        }
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static CNumber[] scalMult(double[] entries, CNumber factor) {
        CNumber[] product = new CNumber[entries.length];

        for(int i=0; i<product.length; i++) {
            product[i] = factor.mult(entries[i]);
        }

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static CNumber[] scalDiv(double[] entries, CNumber divisor) {
        return scalMult(entries, divisor.multInv());
    }


    /**
     * Computes the element-wise multiplication of two tensors. Also called the Hadamard product.
     * @param src1 First tensor in element-wise multiplication.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise multiplication.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise multiplication of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CNumber[] elemMult(CNumber[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        CNumber[] product = new CNumber[src1.length];

        for(int i=0; i<product.length; i++) {
            product[i] = src1[i].mult(src2[i]);
        }

        return product;
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise division of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CNumber[] elemDiv(CNumber[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        CNumber[] product = new CNumber[src1.length];

        for(int i=0; i<product.length; i++) {
            product[i] = src1[i].div(src2[i]);
        }

        return product;
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise division of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CNumber[] elemDiv(double[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        CNumber[] quotient = new CNumber[src1.length];
        double divisor;

        for(int i=0; i<quotient.length; i++) {
            divisor = src2[i].re*src2[i].re + src2[i].im*src2[i].im;
            quotient[i] = new CNumber(src1[i]*src2[i].re / divisor, -src1[i]*src2[i].im / divisor);
        }

        return quotient;
    }
}
