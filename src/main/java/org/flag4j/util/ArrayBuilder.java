/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.util;

import org.flag4j.algebraic_structures.Complex128;

import java.util.Arrays;
import java.util.function.Supplier;


/**
 * <p>The {@code ArrayBuilder} class provides a collection of static utility methods to construct, initialize,
 * and manipulate arrays in various ways. It is designed to simplify array handling tasks,
 * such as creating arrays with default values, filling arrays with specific values, and
 * generating ranges of numbers.
 *
 * <p>This class supports multiple array types, including primitive types ({@code int[]}, {@code double[]}) as well as
 * {@code Complex128} and {@code Complex64}.
 *
 * <h3>Example Usage:</h3>
 * <pre>{@code
 * // Ensure an array exists, if not create an array of size 10.
 * int[] myArray = ArrayBuilder.getOrCreateArray(null, 10);
 *
 * // Create a range of integers.
 * int[] range = ArrayBuilder.intRange(0, 10);
 *
 * // Fill an array with a specific value.
 * int[] filledArray = ArrayBuilder.filledArray(5, 2);
 *
 * // Perform a strided fill of zeros values.
 * double[] stridedArray = new double[10];
 * ArrayBuilder.stridedFillZeros(stridedArray, 2, 3);
 * }</pre>
 *
 * <h3>Restrictions:</h3>
 * <ul>
 *   <li>This class is <i>not</i> designed for jagged (non-rectangular) arrays. All methods dealing with multidimensional arrays
 *   assume, without an explicit check, that all arrays are rectangular.</li>
 * </ul>
 *
 * <p><strong>Note:</strong> This class is a utility class and cannot be instantiated.</p>
 */
public final class ArrayBuilder {

    private ArrayBuilder() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static int[] getOrCreateArray(int[] arr, int size) {
        return arr == null ? new int[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static double[] getOrCreateArray(double[] arr, int size) {
        return arr == null ? new double[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array using the specified {@code initializer} if so.
     * @param arr Array of interest.
     * @param initializer Supplier which constructs a new array of desired size in the case {@code arr==null}.
     * @return If {@code arr == null} then a new array is created by {@code initializer} and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static <T> T[] getOrCreateArray(T[] array, Supplier<T[]> initializer) {
        return array == null ? initializer.get() : array;
    }


    /**
     * Fills an array of complex numbers with zeros.
     *
     * @param dest Array to fill with zeros.
     */
    public static void fillZeros(Complex128[][] dest) {
        for(Complex128[] row : dest)
            Arrays.fill(row, Complex128.ZERO);
    }


    /**
     * <p>
     * Fills an array with zeros separated by the given stride.
     *
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     *
     *
     * @param dest   Array to fill with strided zeros.
     * @param start  Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(double[] dest, int start, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride);
        ValidateParameters.ensureGreaterEq(0, start);

        for(int i = start, stop=dest.length; i < stop; i += stride)
            dest[i] = 0;
    }


    /**
     * <p>
     * Fills an array with zeros separated by the given stride.
     *
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     *
     *
     * @param dest   Array to fill with strided zeros.
     * @param start  Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(Complex128[] dest, int start, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride);
        ValidateParameters.ensureGreaterEq(0, start);

        for(int i = start; i < dest.length; i += stride)
            dest[i] = Complex128.ZERO;
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each separated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified n. Each range of zeros will be separated by
     * the stride.
     *
     *
     * <p>
     * If {@code stride=2}, {@code n=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     *
     *
     * @param dest   Array to fill with strided zeros.
     * @param start  Starting point to apply strided zero fill.
     * @param n Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If {@code stride} or {@code n} is less than one.
     * @throws IllegalArgumentException If {@code start} is less than zero.
     */
    public static void stridedFillZeros(Complex128[] dest, int start, int n, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride, n);
        ValidateParameters.ensureGreaterEq(0, start);
        int step = stride+n;

        for (int i = start; i < dest.length; i += step) {
            for (int j=i, jStop=n+i; j<jStop; j++)
                dest[i] = Complex128.ZERO;
        }
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each separated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified length. Each range of zeros will be separated by
     * the stride.
     *
     *
     * <p>
     * If {@code stride=2}, {@code length=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     *
     *
     * @param dest   Array to fill with strided zeros.
     * @param start  Starting point to apply strided zero fill.
     * @param length Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride or length is less than one.
     * @throws IllegalArgumentException If start is less than zero.
     */
    public static void stridedFillZeros(double[] dest, int start, int length, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride, length);
        ValidateParameters.ensureGreaterEq(0, start);
        int step = stride+length;

        for(int i=start, stop=dest.length; i<stop; i += step) {
            for(int j=i, jStop=length+i; j<jStop; j++)
                dest[i] = 0;
        }
    }


    /**
     * Fills an array with specified value.
     *
     * @param dest      Array to fill.
     * @param fillValue Value to fill array with. This will be converted to a member of the field as if by
     * {@code dest[0].getZero().add(fillValue)}
     */
    public static void fill(Complex128[] dest, double fillValue) {
        Complex128 fillValueComplex = new Complex128(fillValue);
        Arrays.fill(dest, fillValueComplex);
    }


    /**
     * Fills an array with specified value.
     *
     * @param dest      Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static <T> void fill(T[][] dest, T fillValue) {
        for(T[] row : dest)
            Arrays.fill(row, fillValue);
    }


    /**
     * Fills range of an array with specified value.
     *
     * @param dest      Array to fill.
     * @param fillValue Value to fill array with.
     * @param from      Staring index of range (inclusive).
     * @param to        Ending index of range (exclusive).
     */
    public static void fill(Complex128[] dest, double fillValue, int from, int to) {
        ValidateParameters.ensureLessEq(to, from + 1);
        Complex128 complexFillValue = new Complex128(fillValue);
        Arrays.fill(dest, from, to, complexFillValue);
    }


    /**
     * Fills an array with the specified value;
     *
     * @param dest      Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(double[][] dest, double fillValue) {
        for (double[] doubles : dest)
            Arrays.fill(doubles, fillValue);
    }


    /**
     * Constructs an integer array filled with a specific value.
     *
     * @param size  Size of the array.
     * @param value Value to set each index of the array.
     * @return An array of specified {@code size} filled with the specified {@code value}.
     * @throws NegativeArraySizeException If {@code} is negative.
     */
    public static int[] filledArray(int size, int value) {
        int[] dest = new int[size];
        Arrays.fill(dest, value);

        return dest;
    }


    /**
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive)
     *
     * @param start Staring value (inclusive).
     * @param end   Stopping value (exclusive).
     * @return An array containing the integer range {@code [start, end)}.
     * @throws IllegalArgumentException If {@code end < start}.
     */
    public static double[] range(int start, int end) {
        ValidateParameters.ensureGreaterEq(start, end);
        double[] rangeArr = new double[end - start];

        int rangeIdx = 0;
        for(int i = start; i < end; i++)
            rangeArr[rangeIdx++] = i;

        return rangeArr;
    }


    /**
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive)
     *
     * @param start Staring value (inclusive).
     * @param end   Stopping value (exclusive).
     * @return An array containing the integer range {@code [start, end)}.
     * @throws IllegalArgumentException If {@code end < start}.
     */
    public static int[] intRange(int start, int end) {
        ValidateParameters.ensureGreaterEq(start, end);
        int[] rangeArr = new int[end - start];

        int rangeIdx = 0;
        for(int i = start; i < end; i++)
            rangeArr[rangeIdx++] = i;

        return rangeArr;
    }


    /**
     * Gets an array filled with integers from {@code start} (inclusive) to {@code end} (exclusive) where each int is
     * repeated {@code stride} times.
     *
     * @param start  Staring value (inclusive).
     * @param end    Stopping value (exclusive).
     * @param stride Number of times to repeat each integer.
     * @return An array containing the integer range {@code [start, end)} and each integer is repeated {@code stride}
     * times.
     * @throws NegativeArraySizeException If {@code stride} is negative.
     * @throws IllegalArgumentException   If {@code start} is not in {@code [0, end)}
     */
    public static int[] intRange(int start, int end, int stride) {
        ValidateParameters.ensureInRange(start, 0, end, "start");
        int[] rangeArr = new int[(end - start) * stride];

        int k = 0;
        for (int i = start; i < end; i++) {
            Arrays.fill(rangeArr, k, k + stride, i);
            k += stride;
        }

        return rangeArr;
    }
}
