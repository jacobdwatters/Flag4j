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

import java.util.ArrayList;


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
            dest[i] = fillValue.clone();
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
}
