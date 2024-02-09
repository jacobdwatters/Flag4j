/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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
import com.flag4j.io.PrintOptions;

/**
 * A class which provides simple utility methods for {@link String strings}.
 */
public final class StringUtils {

    private StringUtils() { // hide public constructor
        throw new IllegalStateException("Utility class");
    }


    /**
     * Centers a string within a specified size bin.
     * @param s String to center.
     * @param size Size of the bin to center string within.
     * @return A whitespace string of specified size with the string s centered within it.
     */
    public static String center(String s, int size) {
        return center(s, size, " ");
    }


    /**
     * Centers a string within a specified size bin.
     * @param s String to center.
     * @param size Size of the bin to center string within.
     * @param pad Padding character.
     * @return A string made up of the padding character of specified size with the string s centered within it.
     */
    public static String center(String s, int size, String pad) {
        if (s == null || size <= s.length())
            return s;

        StringBuilder sb = new StringBuilder(size);
        sb.append(String.valueOf(pad).repeat((size - s.length()) / 2));
        sb.append(s);
        while (sb.length() < size) {
            sb.append(pad);
        }

        return sb.toString();
    }


    /**
     * Gets the string representation of a double rounded to the specified precision.
     * @param value Value to convert to String.
     * @param precision Precision to round value to.
     * @return The string representation of a double rounded to the specified precision.
     */
    public static String ValueOfRound(double value, int precision) {
        return String.valueOf(CNumber.round(new CNumber(value), PrintOptions.getPrecision()));
    }


    /**
     * Gets the string representation of a {@link CNumber complex number} rounded to the specified precision.
     * @param value Value to convert to String.
     * @param precision Precision to round value to.
     * @return The string representation of a {@link CNumber complex number} rounded to the specified precision.
     */
    public static String ValueOfRound(CNumber value, int precision) {
        return String.valueOf(CNumber.round(value, PrintOptions.getPrecision()));
    }
}
