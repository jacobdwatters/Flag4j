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
public final class FieldOps {

    private FieldOps() {
        // Hide default constructor for utility class.
        throw new IllegalAccessError(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Adds a primitive scalar value to each entry of the {@code src} tensor.
     * @param src Entries of the tensor to add the scalar to.
     * @param scalar Scalar to add to each entry of the tensor.
     * @param dest Array to store the result of adding the scalar to each entry of the tensor. May be {@code null}
     * or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] add(Field<V>[] src, double scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].add(scalar);

        return dest;
    }


    /**
     * Adds a scalar value to each entry of the {@code src} tensor.
     * @param src Entries of the tensor to add the scalar to.
     * @param scalar Scalar to add to each entry of the tensor.
     * @param dest Array to store the result of adding the scalar to each entry of the tensor. May be {@code null}
     * or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] add(double[] src, Field<V> scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalar.add(src[i]);

        return dest;
    }


    /**
     * Subtracts a primitive scalar value from each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to subtract scalar from.
     * @param scalar Scalar to subtract from entry of the tensor.
     * @param dest Array to store the result of subtracting the scalar from each entry of the tensor.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] sub(Field<V>[] src, double scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].sub(scalar);

        return dest;
    }


    /**
     * Subtracts a scalar value from each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to subtract scalar from.
     * @param scalar Scalar to subtract from entry of the tensor.
     * @param dest Array to store the result of subtracting the scalar from each entry of the tensor.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] sub(double[] src, Field<V> scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];
        Field<V> scalarInv = scalar.addInv();

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalarInv.add(src[i]);

        return dest;
    }


    /**
     * Multiplies a primitive scalar value to each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to multiply the scalar to.
     * @param scalar Scalar to multiply to each entry of the tensor.
     * @param dest Array to store the result of multiplying the scalar to each entry of the tensor.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] mult(Field<V>[] src, double scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].mult(scalar);

        return dest;
    }


    /**
     * Multiplies a scalar value to each entry of the {@code src} tensor.
     *
     * @param src Entries of the tensor to multiply the scalar to.
     * @param scalar Scalar to multiply to each entry of the tensor.
     * @param dest Array to store the result of multiplying the scalar to each entry of the tensor.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] mult(double[] src, Field<V> scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalar.mult(src[i]);

        return dest;
    }


    /**
     * Divides each entry of the {@code src} tensor by a scalar.
     * @param src Entries of the tensor.
     * @param scalar Scalar to divide each entry of the tensor by.
     * @param dest Array to store the result of divided each entry of the tensor by the scalar.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] div(Field<V>[] src, V scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].div(scalar);

        return dest;
    }


    /**
     * Divides each entry of the {@code src} tensor by a scalar.
     * @param src Entries of the tensor.
     * @param scalar Scalar to divide each entry of the tensor by.
     * @param dest Array to store the result of divided each entry of the tensor by the scalar.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] div(double[] src, V scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];
        V scalarInv = scalar.multInv();

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = scalarInv.mult(src[i]);

        return dest;
    }


    /**
     * Divides each entry of the {@code src} tensor by a primitive scalar.
     * @param src Entries of the tensor.
     * @param scalar Scalar to divide each entry of the tensor by.
     * @param dest Array to store the result of divided each entry of the tensor by the scalar.
     * May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws ArrayIndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] div(Field<V>[] src, double scalar, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].div(scalar);

        return dest;
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} will be returned. Otherwise, a new array of the appropriate
     * length will be created and returned.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] sqrt(Field<V>[] src, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].sqrt();

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Elements of the tensor.
     * @param factor Factor to scale all elements of {@code src} by.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} a reference to {@code dest} is returned. Otherwise, a new array of the appropriate size will
     * be created and returned.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] scalMult(Field<V>[] src, double factor, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].mult(factor);

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array of the appropriate size
     * is created and returned.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] scalMult(double[] entries, Field<V> factor, Field<V>[] dest) {
        if(dest == null) dest = new Field[entries.length];

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = factor.mult(entries[i]);

        return dest;
    }


    /**
     * Computes the element-wise complex conjugate of a tensor.
     * @param src Entries of the tensor.
     * @param dest Array to store the result in. May be {@code null} or the same array as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array of the appropriate size
     * is created and returned.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] conj(Field<V>[] src, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0; i<src.length; i++)
            dest[i] = (Field) src[i].conj();

        return dest;
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     * @param src Elements of the tensor.
     * @param dest Array to store the result in. May be {@code null} or the same arrays as {@code src}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. Otherwise, a new array of the appropriate size
     * is created and returned.
     * @throws IllegalArgumentException If {@code dest.length < src.length}.
     */
    public static <V extends Field<V>> Field<V>[] recip(Field<V>[] src, Field<V>[] dest) {
        if(dest == null) dest = new Field[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].multInv();

        return dest;
    }


    /**
     * Checks if all elements of a tensor are finite.
     * @param src Elements of the tensor.
     * @return {@code true} if every entry of {@code src} is finite. Otherwise, returns {@code false}.
     */
    public static <V extends Field<V>> boolean isFinite(Field<V>[] src) {
        for(int i=0, size=src.length; i<size; i++)
            if (!src[i].isFinite()) return false;
        return true;
    }


    /**
     * Checks if any element of a tensor is infinite.
     * @param src Elements of the tensor.
     * @return {@code true} if any entry of {@code src} is infinite. Otherwise, returns {@code false}.
     */
    public static <V extends Field<V>> boolean isInfinite(Field<V>[] src) {
        for(int i=0, size=src.length; i<size; i++)
            if (src[i].isInfinite()) return true;
        return false;
    }


    /**
     * Checks if any element of a tensor is NaN.
     * @param src Elements of the tensor.
     * @return {@code true} if any entry of {@code src} is NaN. Otherwise, returns {@code false}.
     */
    public static <V extends Field<V>> boolean isNaN(Field<V>[] src) {
        for(int i=0, size=src.length; i<size; i++)
            if (src[i].isNaN()) return true;
        return false;
    }
}
