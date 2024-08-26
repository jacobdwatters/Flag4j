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

package org.flag4j.operations_old.common.complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ErrorMessages;


/**
 * This class contains low-level implementations for operations_old which check if a complex tensor satisfies some property.
 * Implementations are agnostic to whether the tensor is sparse or dense.
 */
@Deprecated
public final class ComplexProperties {

    private ComplexProperties() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks whether a tensor contains only real values.
     * @param entries Entries of dense tensor or non-zero entries of sparse tensor.
     * @return True if the tensor only contains real values. Returns false otherwise.
     */
    public static boolean isReal(CNumber[] entries) {
        boolean result = true;

        for(CNumber entry : entries) {
            if (entry.im != 0) {
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks whether a tensor contains at least one non-real value.
     * @param entries Entries of dense tensor or non-zero entries of sparse tensor.
     * @return True if the tensor contains at least one non-real value. Returns false otherwise.
     */
    public static boolean isComplex(CNumber[] entries) {
        boolean result = false;

        for(CNumber entry : entries) {
            if(entry.im != 0) {
                result = true;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if an array contains only zeros.
     * @param src Array to check if it only contains zeros.
     * @return True if the {@code src} array contains only zeros.
     */
    public static boolean isZeros(CNumber[] src) {
        boolean allZeros = true;

        for(CNumber value : src) {
            if(value.re!=0 || value.im!=0) {
                allZeros = false;
                break;
            }
        }

        return allZeros;
    }


    /**
     * Checks if all entries of two arrays_old are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays_old have the same length and all entries are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays_old respectively and satisfy
     * {@code |a-b| <= (1E-05 + 1E-08*|b|)}. Otherwise, returns false.
     * @see #allClose(CNumber[], CNumber[], double, double)
     */
    public static boolean allClose(CNumber[] src1, CNumber[] src2) {
        return allClose(src1, src2, 1e-05, 1e-08);
    }



    /**
     * Checks if all entries of two arrays_old are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays_old have the same length and all entries are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays_old respectively and satisfy
     * {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     * @see #allClose(CNumber[], CNumber[])
     */
    public static boolean allClose(CNumber[] src1, CNumber[] src2, double relTol, double absTol) {
        boolean close = src1.length==src2.length;

        if(close) {
            for(int i=0; i<src1.length; i++) {
                double tol = absTol + relTol*src2[i].abs();

                if(src1[i].sub(src2[i]).abs() > tol) {
                    close = false;
                    break;
                }
            }
        }

        return close;
    }
}
