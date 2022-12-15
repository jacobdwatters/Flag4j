/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j.util;

import com.flag4j.complex_numbers.CNumber;

/**
 * This class provides several methods useful for array manipulation.
 */
public final class ArrayUtils {

    private ArrayUtils() {
        // Hide Constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(int[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(double[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(Integer[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(Double[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(String[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Copies an array of {@link CNumber complex numbers}.
     * @param array Array to copy.
     * @return A copy of the specified array.
     */
    public static CNumber[] copyCNumber(CNumber[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }
}
