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

import java.util.*;


/**
 * This class provides several utility methods useful for array manipulation and copying.
 */
@SuppressWarnings("unused")
public final class ArrayUtils {

    private ArrayUtils() {
        // Hide Constructor
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Creates a deep copy of a 2D array.
     * @param src Source array to copy.
     * @param dest Destination array of copy.
     * @throws IllegalArgumentException If the two arrays are not the same shape.
     */
    public static void deepCopy(int[][] src, int[][] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);
        if(src.length > 0) ParameterChecks.assertArrayLengthsEq(src[0].length, dest[0].length);

        for(int i=0; i<src.length; i++) {
            System.arraycopy(src[i], 0, dest[i], 0, src[i].length);
        }
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
    public static void arraycopy(CNumber[] src, int srcPos, CNumber[] dest, int destPos, int length) {
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
    public static void arraycopy(double[] src, int srcPos, CNumber[] dest, int destPos, int length) {
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
            dest[count++] = src[i].copy();
        }

        return dest;
    }


    /**
     * Copies the full array. Similar to {@link Arrays#copyOfRange(Object[], int, int)} but
     * performs a deep copy.
     * @param src Source array to copy from.
     * @return An array of length {@code stop-start} containing a deep copy of the specified range of the source array.
     * @throws NegativeArraySizeException If stop is less than start.
     */
    public static CNumber[] copyOf(CNumber[] src) {
        CNumber[] dest = new CNumber[src.length];

        int count=0;
        for(int i=0; i<dest.length; i++) {
            dest[count++] = src[i].copy();
        }

        return dest;
    }


    /**
     * Fills an array of complex numbers with zeros.
     * @param dest Array to fill with zeros.
     */
    public static void fillZeros(CNumber[] dest) {
        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber();
        }
    }


    /**
     * Fills a specified range of an array of complex numbers with zeros.
     * @param start Starting index of range to fill (inclusive).
     * @param end Ending index of range to fill (Exclusive).
     * @param dest Array to fill specified range with zeros.
     */
    public static void fillZeros(CNumber[] dest, int start, int end) {
        for(int i=start; i<end; i++) {
            dest[i] = new CNumber();
        }
    }


    /**
     * Fills a specified range of an array with zeros.
     * @param start Starting index of range to fill (inclusive).
     * @param end Ending index of range to fill (Exclusive).
     * @param dest Array to fill specified range with zeros.
     */
    public static void fillZeros(double[] dest, int start, int end) {
        System.arraycopy(new double[end-start], 0, dest, start, end-start);
    }


    /**
     * <p>
     * Fills an array with zeros seperated by the given stride.
     * </p>
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     * </p>
     *
     * @param dest Array to fill with strided zeros.
     * @param start Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(double[] dest, int start, int stride) {
        ParameterChecks.assertGreaterEq(1, stride);
        ParameterChecks.assertGreaterEq(0, start);

        for(int i=start; i<dest.length; i+=stride) {
            dest[i] = 0;
        }
    }


    /**
     * <p>
     * Fills an array with zeros seperated by the given stride.
     * </p>
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     * </p>
     *
     * @param dest Array to fill with strided zeros.
     * @param start Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(CNumber[] dest, int start, int stride) {
        ParameterChecks.assertGreaterEq(1, stride);
        ParameterChecks.assertGreaterEq(0, start);

        for(int i=start; i<dest.length; i+=stride) {
            dest[i] = new CNumber();
        }
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each seperated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified length. Each range of zeros will be seperated by
     * the stride.
     * </p>
     *
     * <p>
     * If {@code stride=2}, {@code length=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     * </p>
     *
     * @param dest Array to fill with strided zeros.
     * @param start Starting point to apply strided zero fill.
     * @param length Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride or length is less than one.
     * @throws IllegalArgumentException If start is less than zero.
     */
    public static void stridedFillZeros(CNumber[] dest, int start, int length, int stride) {
        ParameterChecks.assertGreaterEq(1, stride, length);
        ParameterChecks.assertGreaterEq(0, start);

        for(int i=start; i<dest.length; i+=stride+length) {
            for(int j=0; j<length; j++) {
                dest[i+j] = new CNumber();
            }
        }
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each seperated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified length. Each range of zeros will be seperated by
     * the stride.
     * </p>
     *
     * <p>
     * If {@code stride=2}, {@code length=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     * </p>
     *
     * @param dest Array to fill with strided zeros.
     * @param start Starting point to apply strided zero fill.
     * @param length Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride or length is less than one.
     * @throws IllegalArgumentException If start is less than zero.
     */
    public static void stridedFillZeros(double[] dest, int start, int length, int stride) {
        ParameterChecks.assertGreaterEq(1, stride, length);
        ParameterChecks.assertGreaterEq(0, start);

        for(int i=start; i<dest.length; i+=stride+length) {
            for(int j=0; j<length; j++) {
                dest[i+j] = 0;
            }
        }
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(CNumber[] dest, double fillValue) {
        for(int i=0; i<dest.length; i++) {
            dest[i] = new CNumber(fillValue);
        }
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(CNumber[][] dest, CNumber fillValue) {
        for(int i=0; i<dest.length; i++) {
            for(int j=0; j<dest[0].length; j++) {
                dest[i][j] = fillValue.copy();
            }
        }
    }


    /**
     * Fills range of an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     * @param from Staring index of range (inclusive).
     * @param to Ending index of range (exclusive).
     */
    public static void fill(CNumber[] dest, double fillValue, int from, int to) {
        ParameterChecks.assertLessEq(to, from+1);

        for(int i=from; i<to; i++) {
            dest[i] = new CNumber(fillValue);
        }
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(CNumber[] dest, CNumber fillValue) {
        for(int i=0; i<dest.length; i++) {
            dest[i] = fillValue.copy();
        }
    }


    /**
     * Fills an array with specified value.
     * @param dest Array to fill.
     * @param start Starting index of range to fill (Inclusive).
     * @param end Ending index of range to fill (Exclusive).
     * @param fillValue Value to fill array with. Each index of the {@code dest} array will be filled with a deep copy
     *                  of this value.
     * @throws ArrayIndexOutOfBoundsException If {@code start} or {@code end} is not within the destination array.
     */
    public static void fill(CNumber[] dest, int start, int end, CNumber fillValue) {
        for(int i=start; i<end; i++) {
            dest[i] = fillValue.copy();
        }
    }


    /**
     * Fills an array with the specified value;
     * @param dest Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(double[][] dest, double fillValue) {
        for(double[] doubles : dest) {
            Arrays.fill(doubles, fillValue);
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
     * Converts an array of complex numbers to an {@link ArrayList array list}.
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<CNumber> toArrayList(CNumber[] src) {
        ArrayList<CNumber> list = new ArrayList<>(src.length);

        for(CNumber value : src) {
            list.add(value.copy());
        }

        return list;
    }


    /**
     * Converts an array of doubles to a complex {@link ArrayList array list}.
     * @param src Array to convert.
     * @return An equivalent complex array list.
     */
    public static ArrayList<CNumber> toComplexArrayList(double[] src) {
        ArrayList<CNumber> list = new ArrayList<>(src.length);

        for(double value : src) {
            list.add(new CNumber(value));
        }

        return list;
    }


    /**
     * Converts an array of doubles to an {@link ArrayList array list}.
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Integer> toArrayList(int[] src) {
        ArrayList<Integer> list = new ArrayList<>(src.length);

        for(int value : src) {
            list.add(value);
        }

        return list;
    }


    /**
     * Converts a list of {@link Double Doubles} objects to a primitive array.
     * @param src Source list to convert.
     * @return An array containing the same values as the {@code src} list.
     */
    public static double[] fromDoubleList(List<Double> src) {
        double[] dest = new double[src.size()];

        for(int i=0; i<dest.length; i++) {
            dest[i] = src.get(i);
        }

        return dest;
    }



    /**
     * Converts a list of {@link Integer Integer} objects to a primitive array.
     * @param src Source list to convert.
     * @return An array containing the same values as the {@code src} list.
     */
    public static int[] fromIntegerList(List<Integer> src) {
        int[] dest = new int[src.size()];

        for(int i=0; i<dest.length; i++) {
            dest[i] = src.get(i);
        }

        return dest;
    }


    /**
     * Converts a list to an array.
     * @param src Source list to convert.
     * @param dest Destination array to store values from {@code src} in (modified). Must be at least as large as {@code src}.
     * @return A reference to the {@code dest} array.
     * @throws IllegalArgumentException If the {@code dest} array is not large enough to store all entries of {@code src}
     * list.
     */
    public static <T> T[] fromList(List<T> src, T[] dest) {
        ParameterChecks.assertGreaterEq(src.size(), dest.length);
        return src.toArray(dest);
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
     * Swaps elements in an array according to a specified permutation.
     * @param src Array to swap elements within.
     * @param indices Array containing indices of the permutation. If the {@code src} array has length {@code N}, then
     *                the array must be a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If {@code indices} is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void swap(int[] src, int[] indices) {
        ParameterChecks.assertPermutation(indices);

        int[] swapped = new int[src.length];
        int i=0;

        for(int value : indices) {
            swapped[i++] = src[value];
        }

        System.arraycopy(swapped, 0, src, 0, swapped.length);
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
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive)
     * @param start Staring value (inclusive).
     * @param end Stopping value (exclusive).
     * @return An array containing the integer range {@code [start, end)}.
     */
    public static int[] intRange(int start, int end) {
        int[] rangeArr = new int[end-start];

        int j=0;
        for(int i=start; i<end; i++) {
            rangeArr[j++] = i;
        }

        return rangeArr;
    }


    /**
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive) where each int is
     * repeated {@code stride} times.
     * @param start Staring value (inclusive).
     * @param end Stopping value (exclusive).
     * @param stride Number of times to repeat each integer.
     * @return An array containing the integer range {@code [start, end)} and each integer is repeated {@code stride}
     * times.
     * @throws NegativeArraySizeException If {@code stride} is negative.
     * @throws IllegalArgumentException If {@code start} is not in {@code [0, end)}
     */
    public static int[] intRange(int start, int end, int stride) {
        ParameterChecks.assertInRange(start, 0, end, "start");
        int[] rangeArr = new int[(end-start)*stride];

        int k=0;
        for(int i=start; i<end; i++) {
            Arrays.fill(rangeArr, k, k+stride, i);
            k+=stride;
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
     * Checks if a key is in an array.
     * @param src Source array. Must be sorted, if not, call {@link Arrays#sort(double[])} first. Otherwise, the
     *            behavior of this method is undefined.
     * @param key Values to check if they are in the source array.
     * @return A boolean describing if the specified key is in the array or not.
     */
    public static boolean notContains(int[] src, int key) {
        return !contains(src, key);
    }


    /**
     * Checks if a set of keys are in an array.
     * @param src Source array. Must be sorted, if not, call {@link Arrays#sort(double[])} first. Otherwise, the
     *            behavior of this method is undefined.
     * @param keys Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code keys} describing if the associated keys are in the
     * array.
     */
    public static boolean[] contains(double[] src, double... keys) {
        boolean[] result = new boolean[keys.length];

        for(int i=0; i<keys.length; i++) {
            result[i] = contains(src, keys[i]);
        }

        return result;
    }


    /**
     * Checks if a set of keys is in an array.
     * @param src Source array. Must be sorted, if not, call {@link Arrays#sort(int[])} first. Otherwise, the
     *            behavior of this method is undefined.
     * @param keys Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code keys} describing if the associated keys are in the
     * array.
     */
    public static boolean[] contains(int[] src, int... keys) {
        boolean[] result = new boolean[keys.length];

        for(int i=0; i<keys.length; i++) {
            result[i] = contains(src, keys[i]);
        }

        return result;
    }


    /**
     * Checks if an array contains a specified value. This method assumes that the array is sorted as it uses the binary
     * search algorithm. If the array is not sorted, use {@link Arrays#sort(int[])} first.
     * @param arr Array of interest.
     * @param key Value to check for in the {@code arr} array.
     * @return True if the {@code key} value is found in the array. False otherwise.
     * @see Arrays#sort(int[])
     */
    public static boolean contains(int[] arr, int key) {
        return Arrays.binarySearch(arr, key) >= 0;
    }


    /**
     * Checks if an array contains a specified value. This method assumes that the array is sorted as it uses the binary
     * search algorithm. If the array is not sorted, use {@link Arrays#sort(int[])} first.
     * @param arr Array of interest.
     * @param key Value to check for in the {@code arr} array.
     * @return True if the {@code key} value is found in the array. False otherwise.
     * @see Arrays#sort(int[])
     */
    public static boolean contains(double[] arr, double key) {
        return Arrays.binarySearch(arr, key) >= 0;
    }


    /**
     * Flattens a two-dimensional array.
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static int[] flatten(int[][] src) {
        int[] flat = new int[src.length*src[0].length];

        // Copy 2D array to 1D array.
        int i=0;
        for(int[] row : src) {
            for(int value : row) {
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
     * Joins two arrays together.
     * @param src1 First array to join.
     * @param src2 Second array to join.
     * @return A single array of length {@code src1.length + src2.length} containing the elements of {@code src1}
     * followed by the elements of {@code src2}.
     */
    public static double[] join(double[] src1, double[] src2) {
        double[] concatenate = Arrays.copyOf(src1, src1.length + src2.length);
        System.arraycopy(src2, 0, concatenate, src1.length, src2.length);

        return concatenate;
    }


    /**
     * Joins two arrays together.
     * @param src1 First array to join.
     * @param src2 Second array to join.
     * @return A single array of length {@code src1.length + src2.length} containing the elements of {@code src1}
     * followed by the elements of {@code src2}.
     */
    public static int[] join(int[] src1, int[] src2) {
        int[] concatenate = Arrays.copyOf(src1, src1.length + src2.length);
        System.arraycopy(src2, 0, concatenate, src1.length, src2.length);

        return concatenate;
    }


    /**
     * Given a list of integers, {@code srcAxes}, which is a subset of {@code {0, 1, 2, ...., dim-1}}
     * in no particular order, compute the integers which are in {@code {0, 1, 2, ...., dim-1}} but not in
     * {@code srcAxes}.
     * @param srcAxes Source axes which contains a subset of {@code {0, 1, 2, ...., dim-1}} in no particular order.
     * @param dim Dimension of space which contains the axes of interest.
     * @return An array containing the set subtraction {@code {0, 1, 2, ...., dim-1}} - srcAxes.
     */
    public static int[] notinAxes(int[] srcAxes, int dim) {
        int[] notin = new int[dim-srcAxes.length];

        // Copy and sort array.
        int[] srcAxesCopy = srcAxes.clone();
        Arrays.sort(srcAxesCopy);

        int srcIndex = 0;
        int notinIndex = 0;

        for(int i=0; i<dim; i++) {
            if(srcIndex<srcAxesCopy.length && srcAxesCopy[srcIndex]==i) {
                srcIndex++;
            } else {
                notin[notinIndex++] = i;
            }
        }

        return notin;
    }


    /**
     * Shifts all indices in an array by a specified amount.
     * @param shift Amount to shift indices by.
     * @param indices Array of indices to shift.
     * @return A reference to {@code indices}.
     * @see #shiftRange(int, int[], int, int) 
     */
    public static int[] shift(int shift, int[] indices) {
        return shiftRange(shift, indices, 0, indices.length);
    }


    /**
     * Shifts a range of indices in an array by a specified amount.
     * @param shift Amount to shift indices by.
     * @param indices Array of indices to shift.
     * @param start Starting index of range to shift (inclusive).
     * @param stop Stopping index of range to shift (exclusive).
     * @return A reference to {@code indices}.
     * @throws ArrayIndexOutOfBoundsException If start or stop is not within the bounds of the {@code indices} array.
     * @see #shift(int, int[]) 
     */
    public static int[] shiftRange(int shift, int[] indices, int start, int stop) {
        for(int i=start; i<stop; i++) {
            indices[i] += shift;
        }

        return indices;
    }


    /**
     * Gets the unique values from an array and sorts them.
     * @param src The array to get unique values from.
     * @return A sorted array containing all unique values in the {@code src} array.
     */
    public static int[] uniqueSorted(int[] src) {
        HashSet<Integer> hashSet = new HashSet<>();

        for(int j : src) {
            hashSet.add(j);
        }

        return hashSet.stream().mapToInt(Integer::intValue).sorted().toArray();
    }


    /**
     * Finds the fist index of the specified {@code key} within an array. If the element does not exist, then {@code -1}
     * is returned.
     * @param arr Array of interest.
     * @param key Key value to search for.
     * @return Returns the first index of the value {@code key} within the {@code arr} array. If the {@code key} does
     * not occur in the array, {@code -1} will be returned.
     */
    public static int indexOf(int[] arr, int key) {
        for(int i=0; i<arr.length; i++) {
            if(arr[i]==key) {
                return i;
            }
        }

        return -1;
    }


    /**
     * Converts an array of {@link Double} objects to a primitive array (i.e. unboxing).
     * @param arr Array to convert.
     * @return A primitive array equivalent to {@code arr}.
     */
    public static double[] unbox(Double[] arr) {
        int size = arr.length;
        double[] prim = new double[size];

        for(int i=0; i<size; i++) {
            prim[i] = arr[i];
        }

        return prim;
    }


    /**
     * Converts an array of {@link Integer} objects to a primitive array (i.e. unboxing).
     * @param arr Array to convert.
     * @return A primitive array equivalent to {@code arr}.
     */
    public static int[] unbox(Integer[] arr) {
        int size = arr.length;
        int[] prim = new int[size];

        for(int i=0; i<size; i++) {
            prim[i] = arr[i];
        }

        return prim;
    }


    /**
     * Converts a primitive array to an array of equivalent boxed type.
     * @param src The source primitive array to box.
     * @return A boxed array equivalent to the {@code src} primitive array.
     */
    public static Double[] boxed(double[] src) {
        int size = src.length;
        Double[] boxed = new Double[size];

        for(int i=0; i<size; i++) {
            boxed[i] = src[i];
        }

        return boxed;
    }


    /**
     * Converts a primitive array to an array of equivalent boxed type.
     * @param src The source primitive array to box.
     * @return A boxed array equivalent to the {@code src} primitive array.
     */
    public static Integer[] boxed(int[] src) {
        int size = src.length;
        Integer[] boxed = new Integer[size];

        for(int i=0; i<size; i++) {
            boxed[i] = src[i];
        }

        return boxed;
    }


    /**
     * Counts the number of unique elements in an array.
     * @param arr Array to count unique elements in.
     * @return The number of unique elements in {@code arr}.
     */
    public static int numUnique(double[] arr) {
        // For very large arrays, HashMap is quite a bit faster than HashSet.
        Map<Double, Double> map = new HashMap<>(arr.length);

        for(double a : arr) {
            if(!map.containsKey(a)) {
                map.put(a, a);
            }
        }

        return map.keySet().size();
    }


    /**
     * Counts the number of unique elements in an array.
     * @param arr Array to count unique elements in.
     * @return The number of unique elements in {@code arr}.
     */
    public static int numUnique(int[] arr) {
        // For very large arrays, HashMap is quite a bit faster than HashSet.
        Map<Integer, Integer> map = new HashMap<>(arr.length);

        for(int a : arr) {
            if(!map.containsKey(a)) {
                map.put(a, a);
            }
        }

        return map.keySet().size();
    }


    /**
     * Creates a mapping of unique values in {code arr} to integers such that each unique value is mapped to a unique integer and those
     * integers range from {@code 0} to {@link #numUnique(int[]) numUnique(arr) - 1}.
     * @param arr Array to create a mapping for.
     * @return A mapping of unique values in {code arr} to integers such that each unique value is
     * mapped to a unique integer and those integers range from {@code 0} to
     * {@link #numUnique(int[]) numUnique(arr) - 1}.
     */
    @SuppressWarnings("ConstantConditions")
    public static HashMap<Integer, Integer> createUniqueMapping(int[] arr) {
        if (arr.length == 0 || arr == null) return new HashMap<>();

        int[] arrSorted = Arrays.copyOf(arr, arr.length);
        Arrays.sort(arrSorted);

        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(arrSorted[0], 0);

        for(int i=1; i<arrSorted.length; i++) {
            if(arrSorted[i-1]!=arrSorted[i]) {
                map.put(arrSorted[i], ++count);
            }
        }

        return map;
    }


    /**
     * Finds the first and last index of a specified key within a sorted array.
     * @param src The source array to search within. This array is assumed ot be sorted. If the array is not sorted,
     *            call {@link Arrays#sort(int[]) Arrays.sort(src)} before this method. If this is not done, and an
     *            unsorted array is passed to this method, the results are undefined.
     * @param key The key value to find the first and last index of within the {@code src} array.
     * @return An array of length 2 containing the first (inclusive) and last (exclusive) index of the {@code key} within the {@code src} array.
     * If the {@code key} value does not exist in the array, then both first and last index will be
     * {@code (-insertion_point - 1)} where {@code insertion_point} is defined as the index the {@code key} would be
     * inserted into the sorted array.
     */
    public static int[] findFirstLast(int[] src, int key) {
        int keyIdx = Arrays.binarySearch(src, key);

        if(keyIdx < 0) return new int[]{keyIdx, keyIdx}; // Row not found.

        // Find first entry with the specified row key.
        int lowerBound = keyIdx;
        for(int i=keyIdx; i>=0; i--) {
            if(src[i] == key) {
                lowerBound = i;
            } else {
                break;
            }
        }

        int upperBound = keyIdx + 1;
        for(int i=upperBound; i<src.length; i++) {
            if(src[i] == key) {
                upperBound = i + 1;
            } else {
                break;
            }
        }

        return new int[]{lowerBound, upperBound};
    }


    /**
     * Repeats an array a specified number of times.
     * @param numTimes Number of times to repeat the array.
     * @param src The source array to repeat.
     * @return The {@code src} array repeated {@code numTimes times}.
     */
    public static int[] repeat(int numTimes, int[] src) {
        int[] repeated = new int[src.length*numTimes];

        for(int i=0; i<repeated.length; i+=src.length) {
            System.arraycopy(src, 0, repeated, i, src.length);
        }

        return repeated;
    }
}




