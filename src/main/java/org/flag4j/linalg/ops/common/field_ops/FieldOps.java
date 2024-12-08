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

package org.flag4j.linalg.ops.common.field_ops;


import org.flag4j.algebraic_structures.Field;
import org.flag4j.util.ErrorMessages;

/**
 * This utility class contains ops for tensors whose elements are members of a
 * {@link Field}.
 */
public final class FieldOps {

    private FieldOps() {
        // Hide default constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Adds a primitive scalar value to each entry of the {@code src} tensor.
     * @param src Entries of the tensor to add the scalar to.
     * @param scalar Scalar to add to each entry of the tensor.
     * @param dest Array to store the result of adding the scalar to each entry of the tensor. May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void add(V[] src, double scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].add(scalar);
    }


    /**
     * Adds a scalar value to each entry of the {@code src} tensor.
     * @param src Entries of the tensor to add the scalar to.
     * @param scalar Scalar to add to each entry of the tensor.
     * @param dest Array to store the result of adding the scalar to each entry of the tensor. May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void add(double[] src, V scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalar.add(src[i]);
    }


    /**
     * Subtracts a primitive scalar value from each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to subtract scalar from.
     * @param scalar Scalar to subtract from entry of the tensor.
     * @param dest Array to store the result of subtracting the scalar from each entry of the tensor.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void sub(V[] src, double scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].sub(scalar);
    }


    /**
     * Subtracts a scalar value from each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to subtract scalar from.
     * @param scalar Scalar to subtract from entry of the tensor.
     * @param dest Array to store the result of subtracting the scalar from each entry of the tensor.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void sub(double[] src, V scalar, V[] dest) {
        V scalarInv = scalar.addInv();

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalarInv.add(src[i]);
    }


    /**
     * Multiplies a primitive scalar value to each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to multiply the scalar to.
     * @param scalar Scalar to multiply to each entry of the tensor.
     * @param dest Array to store the result of multiplying the scalar to each entry of the tensor.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void mult(V[] src, double scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].mult(scalar);
    }


    /**
     * Multiplies a scalar value to each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to multiply the scalar to.
     * @param scalar Scalar to multiply to each entry of the tensor.
     * @param dest Array to store the result of multiplying the scalar to each entry of the tensor.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void mult(double[] src, V scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalar.mult(src[i]);
    }


    /**
     * Divides each entry of the {@code src} tensor by a scalar.
     * @param src Entries of the tensor.
     * @param scalar Scalar to divide each entry of the tensor by.
     * @param dest Array to store the result of divided each entry of the tensor by the scalar.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void div(V[] src, V scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].div(scalar);
    }


    /**
     * Divides each entry of the {@code src} tensor by a scalar.
     * @param src Entries of the tensor.
     * @param scalar Scalar to divide each entry of the tensor by.
     * @param dest Array to store the result of divided each entry of the tensor by the scalar.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void div(double[] src, V scalar, V[] dest) {
        V scalarInv = scalar.multInv();

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalarInv.mult(src[i]);
    }


    /**
     * Divides each entry of the {@code src} tensor by a primitive scalar.
     * @param src Entries of the tensor.
     * @param scalar Scalar to divide each entry of the tensor by.
     * @param dest Array to store the result of divided each entry of the tensor by the scalar.
     * May be the same array as {@code src}.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void div(V[] src, double scalar, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].div(scalar);
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void sqrt(V[] src, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].sqrt();
    }


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Elements of the tensor.
     * @param factor Factor to scale all elements of {@code src} by.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void scalMult(V[] src, double factor, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].mult(factor);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void scalMult(double[] entries, V factor, V[] dest) {
        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = factor.mult(entries[i]);
    }


    /**
     * Computes the element-wise complex conjugate of a tensor.
     * @param src Entries of the tensor.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void conj(V[] src, V[] dest) {
        for(int i=0; i<src.length; i++)
            dest[i] = src[i].conj();
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @param dest Array to store the result in. May be the same array as {@code src}.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> void recip(V[] src, V[] dest) {
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].multInv();
    }


    /**
     * Checks if all elements of a tensor are finite.
     * @param src Elements of the tensor.
     * @return {@code true} if every entry of {@code src} is finite; {@code false} otherwise.
     */
    public static <V extends Field<V>> boolean isFinite(V[] src) {
        for(int i=0, size=src.length; i<size; i++)
            if (!src[i].isFinite()) return false;

        return true;
    }


    /**
     * Checks if any element of a tensor is infinite.
     * @param src Elements of the tensor.
     * @return {@code true} if any entry of {@code src} is infinite; {@code false} otherwise.
     */
    public static <V extends Field<V>> boolean isInfinite(V[] src) {
        for(int i=0, size=src.length; i<size; i++)
            if (src[i].isInfinite()) return true;

        return false;
    }


    /**
     * Checks if any element of a tensor is NaN.
     * @param src Elements of the tensor.
     * @return {@code true} if any entry of {@code src} is NaN; {@code false} otherwise.
     */
    public static <V extends Field<V>> boolean isNaN(V[] src) {
        for(int i=0, size=src.length; i<size; i++)
            if (src[i].isNaN()) return true;
        return false;
    }
}
