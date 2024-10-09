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
 * This class contains low-level implementations for operations which check if a field tensor satisfies some property.
 * Implementations are agnostic to whether the tensor is sparse or dense.
 */
public final class FieldProperties {

    private FieldProperties() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * Checks if an array contains only zeros.
     * @param src Array to check if it only contains zeros.
     * @return True if the {@code src} array contains only zeros.
     */
    public static <T extends Field<T>> boolean isZeros(Field<T>[] src) {
        for(Field<T> value: src)
            if(!value.isZero()) return false;

        return true;
    }


    /**
     * Checks if all entries of two arrays_old are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays_old have the same length and all entries are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays_old respectively and satisfy
     * {@code |a-b| <= (1E-05 + 1E-08*|b|)}. Otherwise, returns false.
     * @see #allClose(Field[], Field[], double, double)
     */
    public static <T extends Field<T>> boolean allClose(Field<T>[] src1, Field<T>[] src2) {
        return allClose(src1, src2, 1e-05, 1e-08);
    }



    /**
     * Checks if all entries of two arrays_old are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays_old have the same length and all entries are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays_old respectively and satisfy
     * {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     * @see #allClose(Field[], Field[])
     */
    public static <T extends Field<T>> boolean allClose(Field<T>[] src1, Field<T>[] src2, double relTol, double absTol) {
        boolean close = src1.length==src2.length;

        if(close) {
            for(int i=0; i<src1.length; i++) {
                double tol = absTol + relTol*src2[i].abs();

                if(src1[i].sub((T) src2[i]).abs() > tol) {
                    close = false;
                    break;
                }
            }
        }

        return close;
    }
}
