/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;


/**
 * This class provides several methods useful for array manipulation.
 */
public final class ArrayUtils {

    private ArrayUtils() {
        // Hide Constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(int[] src, CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

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
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

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
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

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
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

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
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Performs an array copy similar to {@link System#arraycopy(Object, int, Object, int, int)} but creates a deep copy
     * of each element in the source array.
     * @param src The source array.
     * @param srcPos The starting position from which to copy elements of the source array.
     * @param dest The destination array for the copy.
     * @param destPos Starting index to place copied elements in the destination array.
     * @param length The number of array elements to be copied.
     * @throws ArrayIndexOutOfBoundsException If the destPos parameter plus the length parameter exceeds the length of the
     * source array length or the destination array length.
     */
    public static void arraycopy(@NotNull CNumber[] src, int srcPos, @NotNull CNumber[] dest, int destPos, int length) {
        for(int i=0; i<length; i++) {
            dest[i+destPos] = src[i+srcPos].copy();
        }
    }



    /**
     * Performs an array copy similar to {@link System#arraycopy(Object, int, Object, int, int)} but creates a deep copy
     * of each element in the source array.
     * @param src The source array.
     * @param srcPos The starting position from which to copy elements of the source array.
     * @param dest The destination array for the copy.
     * @param destPos Starting index to place copied elements in the destination array.
     * @param length The number of array elements to be copied.
     * @throws ArrayIndexOutOfBoundsException If the destPos parameter plus the length parameter exceeds the length of the
     * source array length or the destination array length.
     */
    public static void arraycopy(@NotNull double[] src, int srcPos, @NotNull CNumber[] dest, int destPos, int length) {
        for(int i=0; i<length; i++) {
            dest[i+destPos] = new CNumber(src[i+srcPos]);
        }
    }


    /**
     * Copies a range of an array into a new array. Similar to {@link Arrays#copyOfRange(Object[], int, int)} but
     * performs a deep copy.
     * @param src Source array to copy from.
     * @param start Staring index of range to copy (inclusive).
     * @param stop Stopping index of range to copy (Exclusive).
     * @return An array of length {@code stop-start} containing a deep copy of the specified range of the source array.
     * @throws NegativeArraySizeException If stop is less than start.
     */
    public static CNumber[] copyOfRange(CNumber[] src, int start, int stop) {
        CNumber[] dest = new CNumber[stop-start];

        int count=0;
        for(int i=start; i<stop; i++) {
            dest[count++] = src[i];
        }

        return dest;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param src Source array to convert.
     * @param dest Destination array.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     */
    public static void copy2CNumber(CNumber[] src, CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);
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
     * Converts an array of doubles to an {@link ArrayList array list}.
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Double> toArrayList(double[] src) {
        ArrayList<Double> list = new ArrayList<>(src.length);

        for(double value : src) {
            list.add(value);
        }

        return list;
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(CNumber[] dest, CNumber fillValue) {
        // TODO: Investigate speed of using Arrays.setAll(...) and Arrays.parallelSetAll(...)
        for(int i=0; i<dest.length; i++) {
            dest[i] = fillValue.copy();
        }
    }


    /**
     * Fills an array with the specified value;
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(double[][] dest, double fillValue) {
        for (double[] doubles : dest) {
            Arrays.fill(doubles, 1);
        }
    }


    /**
     * Swaps to elements in an array. This is done in place.
     * @param arr Array to swap elements in. This array is modified.
     * @param i Index of first value to swap.
     * @param j Index of second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }



    /**
     * Swaps to elements in an array. This is done in place.
     * @param arr Array to swap elements in. This array is modified.
     * @param i Index of first value to swap.
     * @param j Index of second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(double[] arr, int i, int j) {
        double temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Swaps to elements in an array. This is done in place.
     * @param arr Array to swap elements in. This array is modified.
     * @param i Index of first value to swap.
     * @param j Index of second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(Object[] arr, int i, int j) {
        Object temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive)
     * @param start Staring value (inclusive).
     * @param end Stopping value (exclusive).
     * @return An array containing the integer range {@code [start, end)}.
     */
    public static double[] range(int start, int end) {
        double[] rangeArr = new double[end-start];

        int j=0;
        for(int i=start; i<end; i++) {
            rangeArr[j++]=i;
        }

        return rangeArr;
    }


    /**
     * Checks if a double array is numerically equal to a {@link CNumber complex number} array.
     * @param src1 Double array.
     * @param src2 Complex number array.
     * @return True if all entries in {@code src2} have zero imaginary component and real component equal to the
     * corresponding entry in {@code src1}. Otherwise, returns false.
     */
    public static boolean equals(double[] src1, CNumber[] src2) {
        boolean equal = true;

        if(src1.length != src2.length) {
            equal = false;
        } else {
            for(int i=0; i<src1.length; i++) {
                if(src1[i]!=src2[i].re || src2[i].im != 0) {
                    equal = false;
                    break; // No need to continue.
                }
            }
        }

        return equal;
    }


    /**
     * Checks if a set of values is in an array.
     * @param src Source array.
     * @param values Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code values} describing if the associated values are in the
     * array.
     */
    public static boolean[] inArray(double[] src, double... values) {
        boolean[] result = new boolean[values.length];

        for(double entry : src) {
            for(int i=0; i<values.length; i++) {
                if(entry==values[i]) {
                    result[i]=true;
                }
            }
        }

        return result;
    }


    /**
     * Checks if a value is in an array.
     * @param src Source array.
     * @param value Values to check if they are in the source array.
     * @return A boolean describing if the specified value is in the array or not.
     */
    public static boolean inArray(double[] src, double value) {
        boolean result = false;

        for(double entry : src) {
            if(entry==value) {
                result=true;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if a value is in an array.
     * @param src Source array.
     * @param value Values to check if they are in the source array.
     * @return A boolean describing if the specified value is in the array or not.
     */
    public static boolean inArray(int[] src, int value) {
        boolean result = false;

        for(double entry : src) {
            if(entry==value) {
                result=true;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if a set of values is in an array.
     * @param src Source array.
     * @param values Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code values} describing if the associated values are in the
     * array.
     */
    public static boolean[] inArray(int[] src, int... values) {
        boolean[] result = new boolean[values.length];

        for(double entry : src) {
            for(int i=0; i<values.length; i++) {
                if(entry==values[i]) {
                    result[i]=true;
                }
            }
        }

        return result;
    }


    /**
     * Flattens a two-dimensional array.
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static double[] flatten(double[][] src) {
        double[] flat = new double[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int i=0;
        for(double[] row : src) {
            for(double value : row) {
                flat[i++] = value;
            }
        }

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static CNumber[] flatten(CNumber[][] src) {
        CNumber[] flat = new CNumber[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int i=0;
        for(CNumber[] row : src) {
            for(CNumber value : row) {
                flat[i++] = value.copy();
            }
        }

        return flat;
    }


    /**
     * Constructs a double array from the real components of a complex valued array.
     * @param src Complex array.
     * @return An array containing the real components of the source array.
     */
    public static double[] getReals(CNumber[] src) {
        double[] reals = new double[src.length];

        for(int i=0; i<src.length; i++) {
            reals[i] = src[i].re;
        }

        return reals;
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
                maxLength=currLength;
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
                maxLength=currLength;
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
                maxLength=currLength;
            }
        }

        // Always get last elements' length.
        if(stopIndex < src.length) {
            currLength = CNumber.round(
                    new CNumber(src[src.length-1]),
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength=currLength;
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
                maxLength=currLength;
            }
        }

        // Always get last elements' length.
        if(stopIndex < src.length) {
            currLength = CNumber.round(
                    src[src.length-1],
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength=currLength;
            }
        }

        return maxLength;
    }
}
