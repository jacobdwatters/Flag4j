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

package org.flag4j.linalg.operations.common.semiring_ops;


import org.flag4j.algebraic_structures.semirings.Semiring;

/**
 * This utility class contains operations for tensors whose elements are members of a
 * {@link org.flag4j.algebraic_structures.fields.Field}. The implementations in this class are agnostic
 */
public final class SemiRingOperations {


    /**
     * Computes the scalar multiplication of a tensor with a scalar value.
     * @param src Entries of the tensor.
     * @param factor Scalar value.
     * @return The result of the scalar multiplication of a tensor.
     */
    public static <T extends Semiring<T>> Semiring<T>[] scalMult(Semiring<T>[] src, Semiring<T> factor) {
        return scalMult(src, null, factor);
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param src Entries of the tensor.
     * @param dest Array to store result in. May be null.
     * @param factor Scalar value to multiply.
     * @return A reference to the {@code dest} array if it was not null. Otherwise, a new array will be formed.
     * @throws ArrayIndexOutOfBoundsException If {@code dest} is not at least the size of {@code src}.
     */
    public static <T extends Semiring<T>> Semiring<T>[] scalMult(
            Semiring<T>[] src, Semiring<T>[] dest, Semiring<T> factor) {
        int size = src.length;
        if(dest==null) dest = new Semiring[size];

        for(int i=0; i<size; i++)
            dest[i] = src[i].mult((T) factor);

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
    public static <T extends Semiring<T>> Semiring<T>[] scalMult(
            Semiring<T>[] src, Semiring<T>[] dest, Semiring<T> factor, int start, int stop) {
        if(dest==null) dest = new Semiring[src.length];

        for(int i=start; i<stop; i++)
            dest[i] = src[i].mult((T) factor);

        return dest;
    }


    /**
     * Sums a value to each entry of tensor.
     * @param src Entries of the tensor (non-zero entries if tensor is sparse).
     * @param dest Array to store result in. If {@code null}, an appropriately sized array will be created.
     * @param summand Value to sum to each entry of the tensor.
     * @return A reference to the {@code dest} array if it was not {@code null}. Otherwise, a new array will be formed.
     */
    public static <T extends Semiring<T>> Semiring<T>[] add(Semiring<T>[] src, Semiring<T>[] dest, Semiring<T> summand) {
        if(dest==null) dest = new Semiring[src.length];
        T val = (T) summand;
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].add(val);

        return dest;
    }


    /**
     * Subtracts a value from each entry of tensor.
     * @param src Entries of the tensor (non-zero entries if tensor is sparse).
     * @param dest Array to store result in. If {@code null}, an appropriately sized array will be created.
     * @param subtrahend Value to subtract from each entry of the tensor.
     * @return A reference to the {@code dest} array if it was not {@code null}. Otherwise, a new array will be formed.
     */
    public static <T extends Semiring<T>> Semiring<T>[] sub(Semiring<T>[] src, Semiring<T>[] dest, Semiring<T> subtrahend) {
        if(dest==null) dest = new Semiring[src.length];
        T val = (T) subtrahend;
        for(int i=0, size=src.length; i<size; i++)
            dest[i] = src[i].add(val);

        return dest;
    }
}
