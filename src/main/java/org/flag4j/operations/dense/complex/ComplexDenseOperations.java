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

package org.flag4j.operations.dense.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;


/**
 * This class provides low level methods for computing operations on dense complex tensors.
 */
public final class ComplexDenseOperations {

    private ComplexDenseOperations() {
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
     * @throws IllegalArgumentException If entry arrays_old are not the same size.
     */
    public static Complex128[] add(Field<Complex128>[] src1, Shape shape1, Field<Complex128>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Complex128[] sum = new Complex128[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = src1[i].add((Complex128) src2[i]);

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static Complex128[] add(Field<Complex128>[] src1, Complex128 a) {
        Complex128[] sum = new Complex128[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = src1[i].add(a);

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static Complex128[] add(double[] src1, Complex128 a) {
        Complex128[] sum = new Complex128[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = a.add(src1[i]);

        return sum;
    }


    /**
     * Subtracts a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static Complex128[] sub(double[] src1, Complex128 a) {
        Complex128[] sum = new Complex128[src1.length];
        Complex128 aNeg = a.addInv();

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = aNeg.add(src1[i]);

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
    public static Complex128[] sub(Field<Complex128>[] src1, Shape shape1, Field<Complex128>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Complex128[] diff = new Complex128[src1.length];

        for(int i=0, size=diff.length; i<size; i++)
            diff[i] = src1[i].sub((Complex128) src2[i]);
            
        return diff;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static Complex128[] sub(Field<Complex128>[] src1, Complex128 a) {
        Complex128[] sum = new Complex128[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = src1[i].sub(a);

        return sum;
    }


    /**
     * Computes element-wise subtraction between tensors and stores the result in the first tensor.
     * @param src1 First tensor in subtraction. Also, where the result will be stored.
     * @param shape1 Shape of first tensor.
     * @param src2 Second tensor in the subtraction.
     * @param shape2 Shape of second tensor.
     */
    public static void subEq(Field<Complex128>[] src1, Shape shape1, Field<Complex128>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            src1[i] = src1[i].sub((Complex128) src2[i]);
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static void subEq(Field<Complex128>[] src, Complex128 b) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = src[i].sub(b);
    }


    /**
     * Computes element-wise addition between tensors and stores the result in the first tensor.
     * @param src1 First tensor in addition. Also, where the result will be stored.
     * @param shape1 Shape of first tensor.
     * @param src2 Second tensor in the addition.
     * @param shape2 Shape of second tensor.
     */
    public static void addEq(Field<Complex128>[] src1, Shape shape1, Field<Complex128>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            src1[i] = src1[i].add((Complex128) src2[i]);
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static void addEq(Field<Complex128>[] src, Complex128 b) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = src[i].add(b);
    }


    /**
     * Multiplies all entries in a tensor.
     * @param src Entries of tensor to compute product of.
     * @return The product of all src in the tensor.
     */
    public static Complex128 prod(Field<Complex128>[] src) {
        Complex128 product;

        if(src.length>0) {
            product = new Complex128(1);
            
            for(Field<Complex128> value : src)
               product = product.mult((Complex128) value);
        } else {
            product = Complex128.ZERO;
        }

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static Complex128[] scalMult(Field<Complex128>[] entries, Complex128 a) {
        Complex128[] product = new Complex128[entries.length];

        for(int i=0, size=product.length; i<size; i++)
            product[i] = entries[i].mult(a);

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static Complex128[] scalMult(Field<Complex128>[] entries, double a) {
        Complex128[] product = new Complex128[entries.length];

        for(int i=0, size=product.length; i<size; i++)
            product[i] = entries[i].mult(a);

        return product;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide by.
     * @return The scalar division of the tensor.
     */
    public static Complex128[] scalDiv(Field<Complex128>[] entries, Complex128 divisor) {
        Complex128[] quotient = new Complex128[entries.length];

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = entries[i].div(divisor);

        return quotient;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide by.
     * @return The scalar division of the tensor.
     */
    public static Complex128[] scalDiv(Field<Complex128>[] entries, double divisor) {
        Complex128[] quotient = new Complex128[entries.length];

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = entries[i].div(divisor);

        return quotient;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise reciprocals of the tensor.
     */
    public static Complex128[] recip(Field<Complex128>[] src) {
        Complex128[] recips = new Complex128[src.length];

        for(int i=0, size=recips.length; i<size; i++)
            recips[i] = src[i].multInv();

        return recips;
    }
}
