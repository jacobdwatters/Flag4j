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

package org.flag4j.linalg.operations.common.complex;


import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.util.ErrorMessages;


/**
 * This class provides low level methods for computing operations on complex tensors. These methods can be applied to
 * either sparse or dense complex tensors.
 */
public final class Complex64Operations {

    private Complex64Operations() {
        // Hide constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * Rounds the values of a tensor to the nearest integer. Also see {@link #round(Field<Complex64>[], int)}.
     * @param src Entries of the tensor to round.
     * @return The result of rounding all entries of the source tensor to the nearest integer.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static Complex64[] round(Field<Complex64>[] src) {
        Complex64[] dest = new Complex64[src.length];

        for(int i=0; i<dest.length; i++)
            dest[i] = Complex64.round((Complex64) src[i]);

        return dest;
    }


    /**
     * Rounds the values of a tensor with specified precision. Note, if precision is zero, {@link #round(Field<Complex64>[])} is
     * preferred.
     * @param src Entries of the tensor to round.
     * @param precision Precision to round to (i.e. the number of decimal places).
     * @return The result of rounding all entries of the source tensor with the specified precision.
     * @throws IllegalArgumentException If {@code precision} is negative.
     */
    public static Complex64[] round(Field<Complex64>[] src, int precision) {
        if(precision<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(precision));

        Complex64[] dest = new Complex64[src.length];
        for(int i=0; i<dest.length; i++)
            dest[i] = Complex64.round((Complex64) src[i], precision);

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
    public static Complex64[] roundToZero(Field<Complex64>[] src, float threshold) {
        if(threshold<0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(threshold));

        Complex64[] dest = new Complex64[src.length];

        for(int i=0; i<dest.length; i++)
            dest[i] = Complex64.roundToZero((Complex64) src[i], threshold);

        return dest;
    }
}
