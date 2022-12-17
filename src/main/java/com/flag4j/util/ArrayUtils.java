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
     * @param src Array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(int[] src, CNumber[] dest) {
        ShapeArrayChecks.arrayLengthsCheck(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(double[] src, CNumber[] dest) {
        ShapeArrayChecks.arrayLengthsCheck(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(Integer[] src, CNumber[] dest) {
        ShapeArrayChecks.arrayLengthsCheck(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(Double[] src, CNumber[] dest) {
        ShapeArrayChecks.arrayLengthsCheck(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(String[] src, CNumber[] dest) {
        ShapeArrayChecks.arrayLengthsCheck(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Source array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(CNumber[] src, CNumber[] dest) {
        ShapeArrayChecks.arrayLengthsCheck(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Fills an array with complex numbers with zeros.
     * @param dest Array to fill with zeros.
     */
    public static void fillZeros(CNumber[] dest) {
        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber();
        }
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(CNumber[] dest, double fillValue) {
        // TODO: Investigate speed of using Arrays.setAll(...) and Arrays.parallelSetAll(...)
        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(fillValue);
        }
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(CNumber[] dest, CNumber fillValue) {
        // TODO: Investigate speed of using Arrays.setAll(...) and Arrays.parallelSetAll(...)
        for(int i=0; i<dest.length; i++) {
            dest[i] = fillValue.clone();
        }
    }
}
