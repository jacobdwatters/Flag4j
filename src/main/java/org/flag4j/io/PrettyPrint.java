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

package org.flag4j.io;


import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.StringUtils;

/**
 * Utility class for formatting arrays_old as human-readable strings.
 */
public final class PrettyPrint {


    private PrettyPrint() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Formats array as an abbrivated array so that no more than {@code maxEntries} is actually printed.
     * @param arr Array to format.
     * @param maxEntries The maximum number of entreis to print.
     * @param padding The amount of padding to use between each entry.
     * @param precision The number of decimal places to print for each value.
     * @param centring Flag indicating if each value should be centered within the padding.
     * @return A string representing the abriviated and formatted array.
     */
    public static String abrivatedArray(double[] arr, int maxEntries, int padding, int precision, boolean centring) {
        ParameterChecks.assertNonNegative(maxEntries, padding, precision);

        StringBuilder result = new StringBuilder("[");
        String value;
        int width;

        // Get entries up until the stopping point.
        for(int i=0; i<maxEntries-1 && i<arr.length-1; i++) {
            value = StringUtils.ValueOfRound(arr[i], precision);
            width = padding + value.length();
            value = centring ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Add last value
        if(maxEntries < arr.length) {
            width = padding + 3;
            value = "...";
            value = centring ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        value = StringUtils.ValueOfRound(arr[arr.length-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }


    /**
     * Formats array as an abbrivated array so that no more than {@code maxEntries} is actually printed.
     * @param arr Array to format.
     * @param maxEntries The maximum number of entreis to print.
     * @param padding The amount of padding to use between each entry.
     * @param precision The number of decimal places to print for each value.
     * @param centring Flag indicating if each value should be centered within the padding.
     * @return A string representing the abriviated and formatted array.
     */
    public static String abrivatedArray(CNumber[] arr, int maxEntries, int padding, int precision, boolean centring) {
        ParameterChecks.assertNonNegative(maxEntries, padding, precision);

        StringBuilder result = new StringBuilder("[");
        String value;
        int width;

        // Get entries up until the stopping point.
        for(int i=0; i<maxEntries-1 && i<arr.length-1; i++) {
            value = StringUtils.ValueOfRound(arr[i], precision);
            width = padding + value.length();
            value = centring ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Add last value
        if(maxEntries < arr.length) {
            width = padding + 3;
            value = "...";
            value = centring ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        value = StringUtils.ValueOfRound(arr[arr.length-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }


    /**
     * Formats array as an abbrivated array so that no more than {@code maxCols} is actually printed.
     * @param arr Array to format.
     * @param maxRows Maximum number of rows to print.
     * @param maxCols The maximum number of columns to print.
     * @param padding The amount of padding to use between each entry.
     * @param centring Flag indicating if each value should be centered within the padding.
     * @param offset Offset for array rows after the first.
     * @param centring Flag indicating if each value should be centered within the padding.
     * @return A string representing the abriviated and formatted array.
     */
    public static String abrivatedArray(int[][] arr, int maxRows, int maxCols, int padding, int offset, boolean centering) {
        ParameterChecks.assertNonNegative(maxRows, maxCols, padding);

        StringBuilder result = new StringBuilder("[ ");
        String offsetPad = " ".repeat(offset);
        String value;
        int width;

        // Get entries up until the stopping point.
        for(int i=0, stop=arr.length-1; i<maxRows-1 && i<stop; i++) {
            value = abrivatedArray(arr[i], maxCols, padding, centering);
            width = padding + value.length();
            if(i > 0) result.append(offsetPad);
            result.append(String.format("%-" + width + "s\n", value));
        }

        // Add last value.
        if(maxRows < arr.length) {
            width = padding + 3;
            value = "...";
            result.append(offsetPad).append(" ").append(String.format("%-" + width + "s\n", value));
        }

        value = abrivatedArray(arr[arr.length-1], maxCols, padding, centering);
        width = padding + value.length();
        result.append(offsetPad).append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }


    /**
     * Formats array as an abbrivated array so that no more than {@code maxEntries} is actually printed.
     * @param arr Array to format.
     * @param maxEntries The maximum number of entreis to print.
     * @param padding The amount of padding to use between each entry.
     * @param precision The number of decimal places to print for each value.
     * @param centring Flag indicating if each value should be centered within the padding.
     * @return A string representing the abriviated and formatted array.
     */
    public static String abrivatedArray(int[] arr, int maxEntries, int padding, boolean centring) {
        ParameterChecks.assertNonNegative(maxEntries, padding);

        StringBuilder result = new StringBuilder("[");
        String value;
        int width;

        // Get entries up until the stopping point.
        for(int i=0; i<maxEntries-1 && i<arr.length-1; i++) {
            value = String.valueOf(arr[i]);
            width = padding + value.length();
            value = centring ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Add last value
        if(maxEntries < arr.length) {
            width = padding + 3;
            value = "...";
            value = centring ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        value = String.valueOf(arr[arr.length-1]);
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        return result.append("]").toString();
    }
}
