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

package org.flag4j.complex_numbers;

import org.flag4j.io.PrintOptions;
import org.flag4j.util.ErrorMessages;

/**
 * Contains simple utility functions for the {@link CNumber} object.
 */
@Deprecated
public final class CNumberUtils {

    private CNumberUtils() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles.
     * @param src Array for which to compute the max string representation length of a double in.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(double[] src) {
        int maxLength = -1;
        int currLength;

        for(double value : src) {
            currLength = CNumber.round(new CNumber(value), PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles.
     * @param src Array for which to compute the max string representation length of a double in.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(CNumber[] src) {
        int maxLength = -1;
        int currLength;

        for(CNumber value : src) {
            currLength = CNumber.round(value, PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles up until stopping index.
     * The length of the last element is always considered.
     * @param src Array for which to compute the max string representation length of a double in.
     * @param stopIndex Stopping index for finding max length.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(double[] src, int stopIndex) {
        int maxLength = -1;
        int currLength;

        // Ensure no index out of bound exceptions.
        stopIndex = Math.min(stopIndex, src.length);

        for(int i=0; i<stopIndex; i++) {
            currLength = CNumber.round(
                    new CNumber(src[i]),
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        // Always get last elements' length.
        if(stopIndex < src.length) {
            currLength = CNumber.round(
                    new CNumber(src[src.length-1]),
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles up until stopping index.
     * The length of the last element is always considered.
     * @param src Array for which to compute the max string representation length of a double in.
     * @param stopIndex Stopping index for finding max length.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(CNumber[] src, int stopIndex) {
        int maxLength = -1;
        int currLength;

        // Ensure no index out of bound exceptions.
        stopIndex = Math.min(stopIndex, src.length);

        for(int i=0; i<stopIndex; i++) {
            currLength = CNumber.round(
                    src[i],
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        // Always get last elements' length.
        if(stopIndex < src.length) {
            currLength = CNumber.round(
                    src[src.length-1],
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }
}
