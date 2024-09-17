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

package org.flag4j.operations.dense.real_complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;


/**
 * This class provides low level methods for computing operations with at least one real tensor
 * and at least one complex tensor.
 */
public final class RealComplexDenseOperations {

    private RealComplexDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise addition of two tensors.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src1.length != src2.length}
     */
    public static Complex128[] add(Complex128[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.ensureEqualShape(shape1, shape2);
        Complex128[] sum = new Complex128[src1.length];

        for(int i=0; i<sum.length; i++)
            sum[i] = src1[i].add(src2[i]);

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src1.length != src2.length}
     */
    public static Complex128[] sub(Complex128[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.ensureEqualShape(shape1, shape2);
        Complex128[] diff = new Complex128[src1.length];

        for(int i=0; i<diff.length; i++)
            diff[i] = src1[i].sub(src2[i]);

        return diff;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src1.length != src2.length}
     */
    public static Complex128[] sub(double[] src1, Shape shape1, Complex128[] src2, Shape shape2) {
        ParameterChecks.ensureEqualShape(shape1, shape2);
        Complex128[] diff = new Complex128[src1.length];

        for(int i=0; i<diff.length; i++)
            diff[i] = new Complex128(src1[i]-src2[i].re, -src2[i].im);

        return diff;
    }


    /**
     * Computes element-wise subtraction between tensors and stores the result in the first tensor.
     * @param src1 First tensor in subtraction. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the subtraction.
     * @param shape2 Shape of the second tensor.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src1.length != src2.length}
     */
    public static void subEq(Complex128[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.ensureEqualShape(shape1, shape2);

        for(int i=0; i<src1.length; i++)
            src1[i] = src1[i].sub(src2[i]);
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src TensorOld in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static void subEq(Complex128[] src, double b) {
        for(int i=0; i<src.length; i++)
            src[i] = src[i].sub(b);
    }


    /**
     * Computes element-wise addition between tensors and stores the result in the first tensor.
     * @param src1 First tensor in addition. Also, where the result will be stored.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in the addition.
     * @param shape2 Shape of the second tensor.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src1.length != src2.length}
     */
    public static void addEq(Complex128[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.ensureEqualShape(shape1, shape2);

        for(int i=0; i<src1.length; i++)
            src1[i] = src1[i].add(src2[i]);
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src TensorOld in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static void addEq(Complex128[] src, double b) {
        for(int i=0; i<src.length; i++)
            src[i] = src[i].add(b);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static Complex128[] scalDiv(double[] entries, Complex128 divisor) {
        Complex128[] quotient = new Complex128[entries.length];
        double denom = divisor.re*divisor.re + divisor.im*divisor.im;

        for(int i=0; i<quotient.length; i++) {
            quotient[i] = new Complex128(
                    entries[i]*divisor.re / denom,
                    -entries[i]*divisor.im / denom);
        }

        return quotient;
    }



    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The result of adding the scalar value to all entries of the source tensor.
     */
    public static Complex128[] add(Complex128[] src1, double a) {
        Complex128[] sum = new Complex128[src1.length];

        for(int i=0; i<sum.length; i++)
            sum[i] = src1[i].add(a);

        return sum;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static Complex128[] sub(Complex128[] src1, double a) {
        Complex128[] diff = new Complex128[src1.length];

        for(int i=0, size=diff.length; i<size; i++)
            diff[i] = src1[i].sub(a);

        return diff;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static Complex128[] sub(double[] src1, Complex128 a) {
        Complex128[] diff = new Complex128[src1.length];
        Complex128 aInv = a.addInv();

        for(int i=0; i<diff.length; i++)
            diff[i] = aInv.add(src1[i]);

        return diff;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide tensor by.
     * @return The scalar division of the tensor.
     */
    public static Complex128[] scalDiv(Complex128[] entries, double divisor) {
        Complex128[] quotient = new Complex128[entries.length];

        for(int i=0; i<quotient.length; i++)
            quotient[i] = entries[i].div(divisor);

        return quotient;
    }
}
