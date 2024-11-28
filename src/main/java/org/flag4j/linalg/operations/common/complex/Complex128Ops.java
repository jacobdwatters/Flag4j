/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.operations.common.complex;


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.util.ErrorMessages;

/**
 * This class provides low level methods for computing operations on complex tensors. These methods can be applied to
 * either sparse or dense complex tensors.
 */
public final class Complex128Ops {

    private Complex128Ops() {
        // Hide constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static Complex128[] sqrt(Field<Complex128>[] src) {
        Complex128[] roots = new Complex128[src.length];

        for(int i=0; i<roots.length; i++)
            roots[i] = src[i].sqrt();

        return roots;
    }



    /**
     * Computes the element-wise square root of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise square root of the tensor.
     */
    public static Complex128[] sqrt(double[] src) {
        Complex128[] roots = new Complex128[src.length];

        for(int i=0; i<roots.length; i++)
            roots[i] = Complex128.sqrt(src[i]);

        return roots;
    }


    /**
     * Computes the element-wise absolute value of a tensor.
     * @param src Elements of the tensor.
     * @return The element-wise absolute value of the tensor.
     */
    public static double[] abs(Field<Complex128>[] src) {
        double[] abs = new double[src.length];

        for(int i=0; i<abs.length; i++)
            abs[i] = src[i].mag();

        return abs;
    }


    /**
     * Rounds the values of a tensor to the nearest integer. Also see {@link #round(Complex128[], int)}.
     * @param src Entries of the tensor to round.
     * @return The result of rounding all data of the source tensor to the nearest integer.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static Complex128[] round(Field<Complex128>[] src) {
        Complex128[] dest = new Complex128[src.length];

        for(int i=0; i<dest.length; i++)
            dest[i] = Complex128.round((Complex128) src[i]);

        return dest;
    }


    /**
     * Rounds the values of a tensor with specified precision. Note, if precision is zero, {@link #round(Field[])} is
     * preferred.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e. the number of decimal places).
     * @return The result of rounding all data of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static Complex128[] round(Field<Complex128>[] src, int precision) {
        if(precision<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(precision));

        Complex128[] dest = new Complex128[src.length];
        for(int i=0; i<dest.length; i++)
            dest[i] = Complex128.round((Complex128) src[i], precision);

        return dest;
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this tensor is less than
     *                  the threshold in absolute value then it will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If {@code threshold} is negative.
     */
    public static Complex128[] roundToZero(Field<Complex128>[] src, double threshold) {
        if(threshold<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(threshold));

        Complex128[] dest = new Complex128[src.length];

        for(int i=0; i<dest.length; i++)
            dest[i] = Complex128.roundToZero((Complex128) src[i], threshold);

        return dest;
    }


    /**
     * Computes the scalar multiplication of a tensor.
     * @param entries Entries of the tensor.
     * @param factor Scalar value to multiply.
     * @return The scalar multiplication of the tensor.
     */
    public static Complex128[] scalMult(double[] entries, Complex128 factor) {
        Complex128[] product = new Complex128[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
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
    public static Field<Complex128>[] scalMult(Field<Complex128>[] src, Field<Complex128>[] dest, Complex128 factor) {
        int size = src.length;
        if(dest==null) dest = new Complex128[size];

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
    public static Field<Complex128>[] scalMult(Field<Complex128>[] src, Field<Complex128>[] dest, Complex128 factor, int start, int stop) {
        if(dest==null) dest = new Complex128[src.length];

        for(int i=start; i<stop; i++)
            dest[i] = src[i].mult(factor);

        return dest;
    }


    /**
     * Computes the element-wise complex conjugate of a tensor.
     * @param src Entries of the tensor.
     * @return The element-wise complex conjugate of the tensor
     */
    public static Complex128[] conj(Field<Complex128>[] src) {
        Complex128[] conjugate = new Complex128[src.length];

        for(int i=0; i<src.length; i++)
            conjugate[i] = src[i].conj();

        return conjugate;
    }


    /**
     * Converts a complex tensor to a real tensor by copying the real component and discarding the imaginary component.
     * @param src Entries of complex tensor.
     * @return Equivalent real data for complex tensor.
     */
    public static double[] toReal(Field<Complex128>[] src) {
        double[] real = new double[src.length];

        for(int i=0; i<src.length; i++)
            real[i] = ((Complex128) src[i]).re;

        return real;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide each element ot the tensor by.
     * @return The scalar division of the tensor.
     */
    public static Complex128[] scalDiv(double[] entries, Complex128 divisor) {
        Complex128[] quotient = new Complex128[entries.length];

        double denom = divisor.re*divisor.re + divisor.im*divisor.im;

        for(int i=0, size=entries.length; i<size; i++) {
            double a = entries[i];
            quotient[i] = new Complex128(a*divisor.re / denom, -a*divisor.im / denom);
        }

        return quotient;
    }


    /**
     * Computes the scalar division of a tensor.
     * @param entries Entries of the tensor.
     * @param divisor Scalar value to divide each element ot the tensor by.
     * @return The scalar division of the tensor.
     */
    public static Complex64[] scalDiv(float[] entries, Complex64 divisor) {
        Complex64[] quotient = new Complex64[entries.length];

        float denom = divisor.re*divisor.re + divisor.im*divisor.im;

        for(int i=0, size=entries.length; i<size; i++) {
            float a = entries[i];
            quotient[i] = new Complex64(a*divisor.re / denom, -a*divisor.im / denom);
        }

        return quotient;
    }
}
