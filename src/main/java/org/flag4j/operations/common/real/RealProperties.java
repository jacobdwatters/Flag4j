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

package org.flag4j.operations.common.real;

import org.flag4j.util.ErrorMessages;

/**
 * This class provides low level methods for checking tensor properties. These methods can be applied to
 * either sparse or dense real tensors.
 */
public class RealProperties {

    private RealProperties() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if a tensor only contain positive values. If the tensor is sparse, only the non-zero entries are considered.
     * @param entries Entries of the tensor in question.
     * @return True if the tensor contains only positive values. Otherwise, returns false.
     */
    public static boolean isPos(double[] entries) {
        boolean result = true;

        for(double value : entries) {
            if(value<=0) {
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if a tensor only contain negative values. If the tensor is sparse, only the non-zero entries are considered.
     * @param entries Entries of the tensor in question.
     * @return True if the tensor contains only negative values. Otherwise, returns false.
     */
    public static boolean isNeg(double[] entries) {
        boolean result = true;

        for(double value : entries) {
            if(value>=0) {
                result = false;
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
    public static boolean isZeros(double[] src) {
        boolean allZeros = true;

        for(double value : src) {
            if(value!=0) {
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
     * @see #allClose(double[], double[], double, double)
     */
    public static boolean allClose(double[] src1, double[] src2) {
        return allClose(src1, src2, 1e-05, 1e-08);
    }



    /**
     * Checks if all entries of two arrays_old are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays_old have the same length and all entries are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays_old respectively and satisfy
     * {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     * @see #allClose(double[], double[])
     */
    public static boolean allClose(double[] src1, double[] src2, double relTol, double absTol) {
        boolean close = src1.length==src2.length;

        if(close) {
            for(int i=0; i<src1.length; i++) {
                double tol = absTol + relTol*Math.abs(src2[i]);

                if(Math.abs(src1[i]-src2[i]) > tol) {
                    close = false;
                    break;
                }
            }
        }

        return close;
    }
}
