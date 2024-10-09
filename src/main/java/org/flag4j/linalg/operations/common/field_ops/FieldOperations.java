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

package org.flag4j.linalg.operations.common.field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.util.ErrorMessages;

/**
 * This utility class contains operations for tensors whose elements are members of a
 * {@link org.flag4j.algebraic_structures.fields.Field}.
 */
public final class FieldOperations {

    private FieldOperations() {
        // Hide default constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static Field[] sqrt(Field[] src) {
        Field[] roots = new Field[src.length];
        for(int i=0, size=roots.length; i<size; i++)
            roots[i] = src[i].sqrt();

        return roots;
    }


    /**
     * Computes the element-wise absolute value of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise absolute value of the tensor.
     */
    public static double[] abs(Field[] src) {
        double[] abs = new double[src.length];
        for(int i=0, size=abs.length; i<size; i++)
            abs[i] = src[i].mag();

        return abs;
    }


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Entries of the tensor.
     * @param factor Scalar value.
     * @return The result of the scalar multiplication of a tensor.
     */
    public static Field[] scalMult(Field[] src, double factor) {
        Field[] dest = new Field[src.length];
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].mult(factor);

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Entries of the tensor.
     * @param factor Scalar value.
     * @return The result of the scalar multiplication of a tensor.
     */
    public static Field[] scalMult(Field[] src, Field factor) {
        return scalMult(src, null, factor);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static Field[] scalMult(double[] entries, Field factor) {
        Field[] product = new Field[entries.length];
        for(int i=0; i<product.length; i++)
            product[i] = factor.mult(entries[i]);

        return product;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param src Entries of the tensor.
     * @param dest Array to store result in. May be null.
     * @param factor Scalar value to multiply.
     * @return A reference to the {@code dest} array if it was not null. Otherwise, a new array will be formed.
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not at least the size of {@code src}.
     */
    public static Field[] scalMult(Field[] src, Field[] dest, Field factor) {
        int size = src.length;
        if(dest==null) dest = new Field[size];

        for(int i=0; i<size; i++)
            dest[i] = src[i].mult(factor);

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param src Entries of the tensor.
     * @param dest Array to store result in. May be null.
     * @param factor Scalar value to multiply.
     * @param start Starting index of scalar multiplication.
     * @param stop Stopping index of scalar multiplication.
     * @return A reference to the {@code dest} array if it was not null. Otherwise, a new array will be formed.
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not the size of {@code src}.
     */
    @Deprecated
    public static Field[] scalMult(Field[] src, Field[] dest, Field factor, int start, int stop) {
        if(dest==null) dest = new Field[src.length];

        for(int i=start; i<stop; i++)
            dest[i] = src[i].mult(factor);

        return dest;
    }


    /**
     * Computes the element-wise complex conjugate of a tensor.
     * @param src Entries of the tensor.
     * @return The element-wise complex conjugate of the tensor
     */
    public static Field[] conj(Field[] src) {
        Field[] conjugate = new Field[src.length];
        for(int i=0; i<src.length; i++)
            conjugate[i] = (Field) src[i].conj();

        return conjugate;
    }
}
