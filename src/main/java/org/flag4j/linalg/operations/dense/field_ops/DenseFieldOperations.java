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

package org.flag4j.linalg.operations.dense.field_ops;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;


/**
 * This class provides low level methods for computing operations on dense field tensors.
 */
public final class DenseFieldOperations {

    private DenseFieldOperations() {
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
    public static <T extends Field<T>> Field<T>[] add(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] sum = new Field[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = src1[i].add((T) src2[i]);

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static <T extends Field<T>> Field<T>[] add(Field<T>[] src1, T a) {
        Field<T>[] sum = new Field[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = src1[i].add(a);

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
    public static <T extends Field<T>> Field<T>[] sub(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] diff = new Field[src1.length];

        for(int i=0, size=diff.length; i<size; i++)
            diff[i] = src1[i].sub((T) src2[i]);

        return diff;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static <T extends Field<T>> Field<T>[] sub(Field<T>[] src1, T a) {
        Field<T>[] sum = new Field[src1.length];

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
    public static <T extends Field<T>> void subEq(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            src1[i] = src1[i].sub((T) src2[i]);
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static <T extends Field<T>> void subEq(Field<T>[] src, T b) {
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
    public static <T extends Field<T>> void addEq(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            src1[i] = src1[i].add((T) src2[i]);
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static <T extends Field<T>> void addEq(Field<T>[] src, T b) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = src[i].add(b);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] scalMult(Field<T>[] entries, T a) {
        Field<T>[] product = new Field[entries.length];

        for(int i=0, size=product.length; i<size; i++)
            product[i] = entries[i].mult((T) a);

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor and stores the result in {@code entries}.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     */
    public static <T extends Field<T>> void scalMultEq(Field<T>[] entries, T a) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].mult((T) a);
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide by.
     * @return The scalar division of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] scalDiv(Field<T>[] entries, T divisor) {
        Field<T>[] quotient = new Field[entries.length];

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = entries[i].div((T) divisor);

        return quotient;
    }


    /**
     * Computes the scalar division of a tensor and stores the result in {@code entries}.
     * @param entries Entries of the tensor.
     * @param a Scalar value to divide tensor by.
     */
    public static <T extends Field<T>> void scalDivEq(Field<T>[] entries, T a) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].div((T) a);
    }



    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar addition of the two parameters.
     */
    public static <T extends Field<T>> Field<T>[] add(double[] src1, T a) {
        Field<T>[] sum = new Field[src1.length];

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
    public static <T extends Field<T>> Field<T>[] sub(double[] src1, T a) {
        Field<T>[] sum = new Field[src1.length];
        T aNeg = a.addInv();

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = aNeg.add(src1[i]);

        return sum;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param a Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] scalMult(Field<T>[] entries, double a) {
        Field<T>[] product = new Field[entries.length];

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
    public static <T extends Field<T>> Field<T>[] scalDiv(Field<T>[] entries, double divisor) {
        Field<T>[] quotient = new Field[entries.length];

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = entries[i].div(divisor);

        return quotient;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise reciprocals of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] recip(Field<T>[] src) {
        Field<T>[] recips = new Field[src.length];

        for(int i=0, size=recips.length; i<size; i++)
            recips[i] = src[i].multInv();

        return recips;
    }


    /**
     * Computes the conjugates, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise conjugates of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] conj(Field<T>[] src) {
        Field<T>[] recips = new Field[src.length];

        for(int i=0, size=recips.length; i<size; i++)
            recips[i] = src[i].conj();

        return recips;
    }
}
