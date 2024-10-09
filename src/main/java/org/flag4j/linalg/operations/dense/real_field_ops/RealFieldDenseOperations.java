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

package org.flag4j.linalg.operations.dense.real_field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * This class provides low level methods for computing operations with at least one real tensor
 * and at least one field tensor.
 */
public final class RealFieldDenseOperations {

    private RealFieldDenseOperations() {
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
    public static <T extends Field<T>> Field<T>[] add(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] sum = new Field[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
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
    public static <T extends Field<T>> Field<T>[] sub(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] diff = new Field[src1.length];

        for(int i=0, size=diff.length; i<size; i++)
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
    public static <T extends Field<T>> Field<T>[] sub(double[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] diff = new Field[src1.length];

        for(int i=0, size=diff.length; i<size; i++)
            diff[i] = src2[i].addInv().add(src1[i]);

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
    public static <T extends Field<T>> void subEq(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            src1[i] = src1[i].sub(src2[i]);
    }


    /**
     * Subtracts a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in subtraction. Also, where the result will be stored.
     * @param b Scalar to subtract.
     */
    public static <T extends Field<T>> void subEq(Field<T>[] src, double b) {
        for(int i=0, size=src.length; i<size; i++)
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
    public static <T extends Field<T>> void addEq(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            src1[i] = src1[i].add(src2[i]);
    }


    /**
     * Adds a scalar from each entry of this tensor and stores the result in the tensor.
     * @param src Tensor in addition. Also, where the result will be stored.
     * @param b Scalar to add.
     */
    public static <T extends Field<T>> void addEq(Field<T>[] src, double b) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = src[i].add(b);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] scalDiv(double[] entries, T divisor) {
        Field<T>[] quotient = new Field[entries.length];
        T multInv = divisor.multInv();

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = multInv.mult(entries[i]);

        return quotient;
    }



    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The result of adding the scalar value to all entries of the source tensor.
     */
    public static <T extends Field<T>> Field<T>[] add(Field<T>[] src1, double a) {
        Field<T>[] sum = new Field[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = src1[i].add(a);

        return sum;
    }


    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The result of adding the scalar value to all entries of the source tensor.
     */
    public static <T extends Field<T>> Field<T>[] add(double[] src1, T a) {
        Field<T>[] sum = new Field[src1.length];

        for(int i=0, size=sum.length; i<size; i++)
            sum[i] = a.add(src1[i]);

        return sum;
    }


    /**
     * Subtracts a scalar value from all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The tensor-scalar subtraction of the two parameters.
     */
    public static <T extends Field<T>> Field<T>[] sub(Field<T>[] src1, double a) {
        Field<T>[] diff = new Field[src1.length];

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
    public static <T extends Field<T>> Field<T>[] sub(double[] src1, T a) {
        Field<T>[] diff = new Field[src1.length];
        T aInv = a.addInv();

        for(int i=0, size=diff.length; i<size; i++)
            diff[i] = aInv.add(src1[i]);

        return diff;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide tensor by.
     * @return The scalar division of the tensor.
     */
    public static <T extends Field<T>> Field<T>[] scalDiv(Field<T>[] entries, double divisor) {
        Field<T>[] quotient = new Field[entries.length];

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = entries[i].div(divisor);

        return quotient;
    }
}
