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

package org.flag4j.util;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semirings.Semiring;

import java.util.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;


/**
 * This class provides several utility methods useful for array manipulation and copying.
 */
public final class ArrayUtils {

    // TODO: Class needs to be cleaned up a bit (a lot).

    private ArrayUtils() {
        // Hide Constructor
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static int[] makeNewIfNull(int[] arr, int size) {
        return arr == null ? new int[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static double[] makeNewIfNull(double[] arr, int size) {
        return arr == null ? new double[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static Complex128[] makeNewIfNull(Complex128[] arr, int size) {
        return arr == null ? new Complex128[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static Complex64[] makeNewIfNull(Complex64[] arr, int size) {
        return arr == null ? new Complex64[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static <T extends Semiring<T>> Semiring<T>[] makeNewIfNull(Semiring<T>[] arr, int size) {
        return arr == null ? new Semiring[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static <T extends Ring<T>> Ring<T>[] makeNewIfNull(Ring<T>[] arr, int size) {
        return arr == null ? new Ring[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static <T extends Field<T>> Field<T>[] makeNewIfNull(Field<T>[] arr, int size) {
        return arr == null ? new Field[size] : arr;
    }


    /**
     * Checks if an array is {@code null} and constructs a new array with the specified {@code size} if so.
     * @param arr Array of interest.
     * @param size Size of the array to construct and return in the event that {@code arr == null}.
     * @return If {@code arr == null} then a new array with length {@code size} is created and returned.
     * Otherwise, if {@code arr != null} then a reference to {@code arr} is returned.
     */
    public static Object[] makeNewIfNull(Object[] arr, int size) {
        return arr == null ? new Object[size] : arr;
    }


    /**
     * Computes the cumulative sum of the elements of  an array.
     * @param src Source array to compute cumulative sum within.
     * @param dest Array to store the result of the cumulative sum. May be the same array as {@code src} or {@code null}.
     * @return If {@code dest != null} then a reference to {@code dest} is returned. If {@code dest == null} then a new array of
     * appropriate size will be constructed and returned.
     * @throws IllegalArgumentException If {@code dest != null && dest.length != src.length}.
     */
    public static int[] cumSum(int[] src, int[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);
        dest = makeNewIfNull(dest, src.length);

        for(int i=1, size=src.length; i<size; i++)
            dest[i] = src[i] + src[i-1];

        return dest;
    }


    /**
     * Creates a deep copy of a 2D array. Assumes arrays are <i>not</i> jagged.
     *
     * @param src  Source array to copy.
     * @param dest Destination array of copy. If {@code null}, a new array will be initialized.
     * @return A reference to {@code dest} if it was not {@code null}. In the case where {@code dest} is {@code null}, then a new
     * array will be initialized and returned.
     * @throws IllegalArgumentException If the two arrays are not the same shape.
     */
    public static int[][] deepCopy(int[][] src, int[][] dest) {
        if(dest == null) dest = new int[src.length][src[0].length];
        if(src == dest) return dest;
        else ValidateParameters.ensureGreaterEq(src.length, dest.length);

        for(int i = 0, size=src.length; i < size; i++) {
            dest[i] = new int[src[i].length];
            System.arraycopy(src[i], 0, dest[i], 0, src[i].length);
        }

        return dest;
    }


    /**
     * Converts array to an array of {@link Complex128 complex numbers}.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] wrapAsComplex128(int[] src, Complex128[] dest) {
        dest = makeNewIfNull(dest, src.length);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts array to an array of {@link Complex128 complex numbers}.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is {@code null}, a new array will be created.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] wrapAsComplex128(double[] src, Complex128[] dest) {
        dest = makeNewIfNull(dest, src.length);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts array to an array of {@link Complex128 complex numbers}.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] wrapAsComplex128(Integer[] src, Complex128[] dest) {
        dest = makeNewIfNull(dest, src.length);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts array to an array of {@link Complex128 complex numbers}.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] wrapAsComplex128(Double[] src, Complex128[] dest) {
        dest = makeNewIfNull(dest, src.length);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts an array to an array of {@link Complex128 complex numbers}.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] wrapAsComplex128(Complex64[] src, Complex128[] dest) {
        dest = makeNewIfNull(dest, src.length);
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Converts array to an array of {@link Complex128 complex numbers}.
     *
     * @param src  Array to convert.
     * @param dest Destination array. If the destination array is null, a new array will be created.
     * @throws IllegalArgumentException If source and destination arrays do not have the same length.
     * @return A reference to the {@code dest} array.
     */
    public static Complex128[] wrapAsComplex128(String[] src, Complex128[] dest) {
        if(dest == null) {
            dest = new Complex128[src.length];
        }
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0, size=dest.length; i<size; i++)
            dest[i] = new Complex128(src[i]);

        return dest;
    }


    /**
     * Checks if two primitive 2D integer arrays are element-wise equal.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return The
     */
    public static boolean deepEquals(int[][] src1, int[][] src2) {
        if(src1 == src2) return true;
        if(src1 == null || src2 == null) return false;
        if(src1.length != src2.length) return false;

        for(int i=0, size=src1.length; i<size; i++)
            if(!Arrays.equals(src1[i], src2[i])) return false;

        return true;
    }


    /**
     * Performs an array copy similar to {@link System#arraycopy(Object, int, Object, int, int)} but wraps doubles as complex numbers.
     *
     * @param src     The source array.
     * @param srcPos  The starting position from which to copy elements of the source array.
     * @param dest    The destination array for the copy.
     * @param destPos Starting index to place copied elements in the destination array.
     * @param length  The number of array elements to be copied.
     * @throws ArrayIndexOutOfBoundsException If the destPos parameter plus the length parameter exceeds the length of the
     *                                        source array length or the destination array length.
     */
    public static void arraycopy(double[] src, int srcPos, Field<Complex128>[] dest, int destPos, int length) {
        for(int i = 0; i < length; i++)
            dest[i + destPos] = new Complex128(src[i + srcPos]);
    }



    /**
     * Fills an array of complex numbers with zeros.
     *
     * @param dest Array to fill with zeros.
     */
    public static void fillZeros(Field<Complex128>[][] dest) {
        for(Field<Complex128>[] row : dest)
            Arrays.fill(row, Complex128.ZERO);
    }


    /**
     * <p>
     * Fills an array with zeros separated by the given stride.
     * </p>
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     * </p>
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
     * </p>
     *
     * <p>
     * If {@code stride=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}} then the result will be {@code {1, 0, 3, 4, 0, 6, 7, 0, 9}}.
     * </p>
     *
     * @param dest   Array to fill with strided zeros.
     * @param start  Staring point in array to apply strided zero fill.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride is less than 1.
     * @throws IllegalArgumentException If start is less than 0.
     */
    public static void stridedFillZeros(Field<Complex128>[] dest, int start, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride);
        ValidateParameters.ensureGreaterEq(0, start);

        for(int i = start; i < dest.length; i += stride)
            dest[i] = Complex128.ZERO;
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each separated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified length. Each range of zeros will be separated by
     * the stride.
     * </p>
     *
     * <p>
     * If {@code stride=2}, {@code length=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     * </p>
     *
     * @param dest   Array to fill with strided zeros.
     * @param start  Starting point to apply strided zero fill.
     * @param length Number of sequential zeros to fill per stride.
     * @param stride Number of elements between each value to set to zero within the destination array.
     * @throws IllegalArgumentException If stride or length is less than one.
     * @throws IllegalArgumentException If start is less than zero.
     */
    public static void stridedFillZeros(Field<Complex128>[] dest, int start, int length, int stride) {
        ValidateParameters.ensureGreaterEq(1, stride, length);
        ValidateParameters.ensureGreaterEq(0, start);
        int step = stride+length;

        for (int i = start; i < dest.length; i += step) {
            for (int j=i, jStop=length+i; j<jStop; j++)
                dest[i] = Complex128.ZERO;
        }
    }


    /**
     * <p>
     * Fills an array with a range of zeros, each separated by the given stride. Specifically, the destination array will
     * be filled with several sequential ranges of zeros of specified length. Each range of zeros will be separated by
     * the stride.
     * </p>
     *
     * <p>
     * If {@code stride=2}, {@code length=3}, {@code start=1}, and {@code dest={1, 2, 3, 4, 5, 6, 7, 8, 9}}
     * then the result will be {1, 0, 0, 0, 5, 0, 0, 0, 9}.
     * </p>
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
    public static void fill(Field<Complex128>[] dest, double fillValue) {
        Complex128 fillValueComplex = new Complex128(fillValue);
        Arrays.fill(dest, fillValueComplex);
    }


    /**
     * Fills an array with specified value.
     *
     * @param dest      Array to fill.
     * @param fillValue Value to fill array with.
     */
    public static void fill(Field<Complex128>[][] dest, Complex128 fillValue) {
        for(Field<Complex128>[] row : dest)
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
    public static void fill(Field<Complex128>[] dest, double fillValue, int from, int to) {
        ValidateParameters.ensureLessEq(to, from + 1);
        Complex128 complexFillValue = new Complex128(fillValue);
        Arrays.fill(dest, from, to, complexFillValue);
    }


    /**
     * Fills an array with specified value. Similar to {@link Arrays#fill(Object[], Object)} but creates  deep copy of the fill value
     * for each position.
     *
     * @param dest      Array to fill.
     * @param start     Starting index of range to fill (Inclusive).
     * @param end       Ending index of range to fill (Exclusive).
     * @param fillValue Value to fill array with. Each index of the {@code dest} array will be filled with a deep copy
     *                  of this value.
     * @throws ArrayIndexOutOfBoundsException If {@code start} or {@code end} is not within the destination array.
     */
    public static void fill(Field<Complex128>[] dest, int start, int end, Complex128 fillValue) {
        Arrays.fill(dest, start, end, fillValue);
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
     * Converts an array of doubles to an {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Double> toArrayList(double[] src) {
        ArrayList<Double> list = new ArrayList<>(src.length);

        for(double value : src)
            list.add(value);

        return list;
    }


    /**
     * Converts an array of complex numbers to an {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Field<Complex128>> toArrayList(Field<Complex128>[] src) {
        ArrayList<Field<Complex128>> list = new ArrayList<>(src.length);
        list.addAll(Arrays.asList(src));
        return list;
    }


    /**
     * Converts an array of doubles to a complex {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent complex array list.
     */
    public static ArrayList<Field<Complex128>> toComplexArrayList(double[] src) {
        ArrayList<Field<Complex128>> list = new ArrayList<>(src.length);

        for (double value : src)
            list.add(new Complex128(value));

        return list;
    }


    /**
     * Converts an array of doubles to an {@link ArrayList array list}.
     *
     * @param src Array to convert.
     * @return An equivalent array list.
     */
    public static ArrayList<Integer> toArrayList(int[] src) {
        ArrayList<Integer> list = new ArrayList<>(src.length);

        for (int value : src)
            list.add(value);

        return list;
    }


    /**
     * Converts a list of {@link Double Doubles} objects to a primitive array.
     *
     * @param src Source list to convert.
     * @return An array containing the same values as the {@code src} list.
     */
    public static double[] fromDoubleList(List<Double> src) {
        double[] dest = new double[src.size()];

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = src.get(i);

        return dest;
    }


    /**
     * Converts a list of {@link Integer Integer} objects to a primitive array.
     *
     * @param src Source list to convert.
     * @return An array containing the same values as the {@code src} list.
     */
    public static int[] fromIntegerList(List<Integer> src) {
        int[] dest = new int[src.size()];

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = src.get(i);

        return dest;
    }


    /**
     * Converts a list of {@link Integer Integer} objects to a primitive array.
     *
     * @param src  Source list to convert.
     * @param dest Destination array to store values from {@code src} in (modified). Must be at least as large as {@code src}.
     * @return A reference to the {@code dest} array.
     */
    public static int[] fromIntegerList(List<Integer> src, int[] dest) {
        ValidateParameters.ensureGreaterEq(src.size(), dest.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = src.get(i);

        return dest;
    }


    /**
     * Converts a list to an array.
     *
     * @param src  Source list to convert.
     * @param dest Destination array to store values from {@code src} in (modified). Must be at least as large as {@code src}.
     * @return A reference to the {@code dest} array.
     * @throws IllegalArgumentException If the {@code dest} array is not large enough to store all data of {@code src}
     *                                  list.
     */
    public static <T> T[] fromList(List<T> src, T[] dest) {
        ValidateParameters.ensureGreaterEq(src.size(), dest.length);
        return src.toArray(dest);
    }


    /**
     * Swaps to elements in an array. This is done in place.
     *
     * @param arr Array to swap elements in. This array is modified.
     * @param i   Index of first value to swap.
     * @param j   Index of second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Swaps elements in an array according to a specified permutation.
     *
     * @param src     Array to swap elements within.
     * @param indices Array containing indices of the permutation. If the {@code src} array has length {@code N}, then
     *                the array must be a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @throws IllegalArgumentException If {@code indices} is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void swap(int[] src, int[] indices) {
        ValidateParameters.ensurePermutation(indices);

        int[] swapped = new int[src.length];
        int i = 0;

        for(int value : indices)
            swapped[i++] = src[value];

        System.arraycopy(swapped, 0, src, 0, swapped.length);
    }


    /**
     * Swaps elements in an array according to a specified permutation. This method should be used with extreme caution as unlike
     * {@link #swap(int[], int[])}, this method does <i>not</i> verify that {@code indices} is a permutation.
     *
     * @param src     Array to swap elements within.
     * @param indices Array containing indices of the permutation. If the {@code src} array has length {@code N}, then
     *                the array must be a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void swapUnsafe(int[] src, int[] indices) {
        int[] swapped = new int[src.length];
        int i = 0;

        for(int value : indices)
            swapped[i++] = src[value];

        System.arraycopy(swapped, 0, src, 0, swapped.length);
    }


    /**
     * Swaps to elements in an array. This is done in place.
     *
     * @param arr Array to swap elements in. This array is modified.
     * @param i   Index of first value to swap.
     * @param j   Index of second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(double[] arr, int i, int j) {
        double temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }


    /**
     * Swaps to elements in an array. This is done in place.
     *
     * @param arr Array to swap elements in. This array is modified.
     * @param i   Index of first value to swap.
     * @param j   Index of second value to swap.
     * @throws IndexOutOfBoundsException If {@code i} or {@code j} are out of the bounds of {@code arr}.
     */
    public static void swap(Object[] arr, int i, int j) {
        Object temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
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


    /**
     * Checks if a double array is numerically equal to a {@link Complex128 complex number} array.
     *
     * @param src1 Double array.
     * @param src2 Complex number array.
     * @return True if all data in {@code src2} have zero imaginary component and real component equal to the
     * corresponding entry in {@code src1}. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean equals(double[] src1, Field<T>[] src2) {
        boolean equal = true;

        if (src1.length != src2.length) {
            equal = false;
        } else {
            for(int i=0, size = src1.length; i < size; i++) {
                if (src1[i] != ((Complex128) src2[i]).re || ((Complex128) src2[i]).im != 0) {
                    equal = false;
                    break; // No need to continue.
                }
            }
        }

        return equal;
    }


    /**
     * Checks if a key is in an array.
     *
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
     *
     * @param src  Source array. Must be sorted, if not, call {@link Arrays#sort(double[])} first. Otherwise, the
     *             behavior of this method is undefined.
     * @param keys Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code keys} describing if the associated keys are in the
     * array.
     */
    public static boolean[] contains(double[] src, double... keys) {
        boolean[] result = new boolean[keys.length];

        for (int i = 0, size= keys.length; i < size; i++) {
            result[i] = contains(src, keys[i]);
        }

        return result;
    }


    /**
     * Checks if a set of keys is in an array.
     *
     * @param src  Source array. Must be sorted, if not, call {@link Arrays#sort(int[])} first. Otherwise, the
     *             behavior of this method is undefined.
     * @param keys Values to check if they are in the source array.
     * @return A boolean array with the same length as {@code keys} describing if the associated keys are in the
     * array.
     */
    public static boolean[] contains(int[] src, int... keys) {
        boolean[] result = new boolean[keys.length];

        for (int i = 0, size=keys.length; i < size; i++) {
            result[i] = contains(src, keys[i]);
        }

        return result;
    }


    /**
     * Checks if an array contains a specified value. This method assumes that the array is sorted as it uses the binary
     * search algorithm. If the array is not sorted, use {@link Arrays#sort(int[])} first.
     *
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
     *
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
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static int[] flatten(int[][] src) {
        int[] flat = new int[src.length * src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for(int[] row : src)
            for (int value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static double[] flatten(double[][] src) {
        double[] flat = new double[src.length * src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (double[] row : src)
            for (double value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array and unboxes.
     *
     * @param src Array to flatten and unbox.
     * @return The flattened array.
     */
    public static double[] unboxFlatten(Double[][] src) {
        double[] flat = new double[src.length * src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (Double[] row : src)
            for (double value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static Complex128[] flatten(Complex128[][] src) {
        Complex128[] flat = new Complex128[src.length * src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (Complex128[] row : src)
            for (Complex128 value : row)
                flat[flatIdx++] = value;

        return flat;
    }


    /**
     * Flattens a two-dimensional array.
     *
     * @param src Array to flatten.
     * @return The flattened array.
     */
    public static <T extends Field<T>> Field<T>[] flatten(Field<T>[][] src) {
        Field<T>[] flat = new Field[src.length * src[0].length];

        // Copy 2D array to 1D array.
        int flatIdx = 0;
        for (Field<T>[] row : src)
            for (Field<T> value : row)
                flat[flatIdx++] = value;

        return (T[]) flat;
    }


    /**
     * Joins two arrays together.
     *
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
     *
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
     *
     * @param srcAxes Source axes which contains a subset of {@code {0, 1, 2, ...., dim-1}} in no particular order.
     * @param dim     Dimension of space which contains the axes of interest.
     * @return An array containing the set subtraction {@code {0, 1, 2, ...., dim-1}} - srcAxes.
     */
    public static int[] notInAxes(int[] srcAxes, int dim) {
        int[] notIn = new int[dim - srcAxes.length];

        // Copy and sort array.
        int[] srcAxesCopy = srcAxes.clone();
        Arrays.sort(srcAxesCopy);

        int srcIndex = 0;
        int notInIndex = 0;

        for (int i = 0; i < dim; i++) {
            if(srcIndex < srcAxesCopy.length && srcAxesCopy[srcIndex] == i)
                srcIndex++;
            else
                notIn[notInIndex++] = i;
        }

        return notIn;
    }


    /**
     * Shifts all indices in an array by a specified amount.
     *
     * @param shift   Amount to shift indices by.
     * @param indices Array of indices to shift.
     * @return A reference to {@code indices}.
     * @see #shiftRange(int, int[], int, int)
     */
    public static int[] shift(int shift, int[] indices) {
        return shiftRange(shift, indices, 0, indices.length);
    }


    /**
     * Shifts a range of indices in an array by a specified amount.
     *
     * @param shift   Amount to shift indices by.
     * @param indices Array of indices to shift.
     * @param start   Starting index of range to shift (inclusive).
     * @param stop    Stopping index of range to shift (exclusive).
     * @return A reference to {@code indices}.
     * @throws ArrayIndexOutOfBoundsException If start or stop is not within the bounds of the {@code indices} array.
     * @see #shift(int, int[])
     */
    public static int[] shiftRange(int shift, int[] indices, int start, int stop) {
        for (int i = start; i < stop; i++)
            indices[i] += shift;

        return indices;
    }


    /**
     * Gets the unique values from an array and sorts them.
     *
     * @param src The array to get unique values from.
     * @return A sorted array containing all unique values in the {@code src} array.
     */
    public static int[] uniqueSorted(int[] src) {
        HashSet<Integer> hashSet = new HashSet<>();

        for(int j : src)
            hashSet.add(j);

        return hashSet.stream().mapToInt(Integer::intValue).sorted().toArray();
    }


    /**
     * Finds the fist index of the specified {@code key} within an array. If the element does not exist, then {@code -1}
     * is returned.
     *
     * @param arr Array of interest.
     * @param key Key value to search for.
     * @return Returns the first index of the value {@code key} within the {@code arr} array. If the {@code key} does
     * not occur in the array, {@code -1} will be returned.
     */
    public static int indexOf(int[] arr, int key) {
        for(int i = 0, size=arr.length; i < size; i++)
            if (arr[i] == key) return i;

        return -1;
    }


    /**
     * Converts an array of {@link Double} objects to a primitive array (i.e. unboxing).
     *
     * @param arr Array to unbox.
     * @param dest Destination array for the unboxed values.
     * @return If dest was not {@code null} then a reference to {@code dest} is returned. Otherwise, a new array with the unboxed
     * values is returned.
     */
    public static double[] unbox(Double[] arr, double[] dest) {
        dest = makeNewIfNull(dest, arr.length);

        for (int i=0, size=dest.length; i<size; i++)
            dest[i] = arr[i];

        return dest;
    }


    /**
     * Converts an array of {@link Integer} objects to a primitive array (i.e. unboxing).
     * @param arr Array to unbox.
     * @param dest Destination array for the unboxed values.
     * @return If dest was not {@code null} then a reference to {@code dest} is returned. Otherwise, a new array with the unboxed
     * values is returned.
     */
    public static int[] unbox(Integer[] arr, int[] dest) {
        int size = arr.length;
        int[] prim = new int[size];

        for (int i = 0; i < size; i++)
            prim[i] = arr[i];

        return prim;
    }


    /**
     * Converts a primitive array to an array of equivalent boxed type.
     *
     * @param src The source primitive array to box.
     * @return A boxed array equivalent to the {@code src} primitive array.
     */
    public static Double[] boxed(double[] src) {
        int size = src.length;
        Double[] boxed = new Double[size];

        for (int i = 0; i < size; i++)
            boxed[i] = src[i];

        return boxed;
    }


    /**
     * Converts a primitive array to an array of equivalent boxed type.
     *
     * @param src The source primitive array to box.
     * @return A boxed array equivalent to the {@code src} primitive array.
     */
    public static Integer[] boxed(int[] src) {
        int size = src.length;
        Integer[] boxed = new Integer[size];

        for (int i = 0; i < size; i++)
            boxed[i] = src[i];

        return boxed;
    }


    /**
     * Counts the number of unique elements in an array.
     *
     * @param arr Array to count unique elements in.
     * @return The number of unique elements in {@code arr}.
     */
    public static int numUnique(double[] arr) {
        // For very large arrays, HashMap is quite a bit faster than HashSet.
        Map<Double, Double> map = new HashMap<>(arr.length);

        for(double a : arr)
            if(!map.containsKey(a)) map.put(a, a);

        return map.keySet().size();
    }


    /**
     * Counts the number of unique elements in an array.
     *
     * @param arr Array to count unique elements in.
     * @return The number of unique elements in {@code arr}.
     */
    public static int numUnique(int[] arr) {
        // For very large arrays, HashMap is quite a bit faster than HashSet.
        Map<Integer, Integer> map = new HashMap<>(arr.length);

        for(int a : arr)
            if (!map.containsKey(a)) map.put(a, a);

        return map.keySet().size();
    }


    /**
     * Creates a mapping of unique values in {code arr} to integers such that each unique value is mapped to a unique integer and those
     * integers range from {@code 0} to {@link #numUnique(int[]) numUnique(arr) - 1}.
     *
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

        for (int i = 1; i < arrSorted.length; i++) {
            if (arrSorted[i - 1] != arrSorted[i]) {
                map.put(arrSorted[i], ++count);
            }
        }

        return map;
    }


    /**
     * Finds the first and last index of a specified key within a sorted array.
     *
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

        if (keyIdx < 0) return new int[]{keyIdx, keyIdx}; // Row not found.

        // Find first entry with the specified row key.
        int lowerBound = keyIdx;
        for (int i = keyIdx; i >= 0; i--) {
            if (src[i] == key) lowerBound = i;
            else break;
        }

        int upperBound = keyIdx + 1;
        for (int i = upperBound; i < src.length; i++) {
            if (src[i] == key) upperBound = i + 1;
            else break;
        }

        return new int[]{lowerBound, upperBound};
    }


    /**
     * Repeats an array a specified number of times.
     *
     * @param numTimes Number of times to repeat the array.
     * @param src      The source array to repeat.
     * @return The {@code src} array repeated {@code numTimes times}.
     */
    public static int[] repeat(int numTimes, int[] src) {
        int[] repeated = new int[src.length * numTimes];

        for(int i = 0, size=repeated.length, step=src.length; i < size; i += step)
            System.arraycopy(src, 0, repeated, i, src.length);

        return repeated;
    }


    /**
     * Constructs an array filled with a specific value.
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
     * Converts an array of ints to an array of doubles.
     *
     * @param src  Source array to convert.
     * @param dest Destination array to store double values equivalent to the values in the {@code src} array.
     *             If null, a new double array with the same size as {@code src} will be created.
     * @return A reference to the {@code dest} array.
     */
    public static double[] asDouble(int[] src, double[] dest) {
        dest = makeNewIfNull(dest, src.length);

        for (int i = 0, size=src.length; i < size; i++)
            dest[i] = src[i];

        return dest;
    }


    /**
     * Converts an array of {@link Integer Integers} to an array of doubles.
     *
     * @param src  Source array to convert.
     * @param dest Destination array to store double values equivalent to the values in the {@code src} array.
     *             If null, a new double array with the same size as {@code src} will be created.
     * @return A reference to the {@code dest} array.
     */
    public static double[] asDouble(Integer[] src, double[] dest) {
        if (dest == null) dest = new double[src.length];

        for(int i = 0, size=src.length; i < size; i++)
            dest[i] = src[i];

        return dest;
    }


    /**
     * Splices an array into another array at the specified index.
     *
     * @param arr1      First array.
     * @param arr2      Array to splice into {@code arr1}.
     * @param spliceIdx The index within {@code arr1} to splice {@code arr2} into.
     * @return The result of splicing {@code arr2} into {@code arr1} at the index {@code spliceIdx}.
     */
    public static Complex128[] splice(Field<Complex128>[] arr1, Field<Complex128>[] arr2, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(arr1.length + 1, spliceIdx);
        Complex128[] spliced = new Complex128[arr1.length + arr2.length];

        System.arraycopy(arr1, 0, spliced, 0, spliceIdx);
        System.arraycopy(arr2, 0, spliced, spliceIdx, arr2.length);
        System.arraycopy(arr1, spliceIdx, spliced, spliceIdx + arr2.length, arr1.length - spliceIdx);

        return spliced;
    }


    /**
     * Splices an array into another array at the specified index.
     *
     * @param arr1      First array.
     * @param arr2      Array to splice into {@code arr1}.
     * @param spliceIdx The index within {@code arr1} to splice {@code arr2} into.
     * @return The result of splicing {@code arr2} into {@code arr1} at the index {@code spliceIdx}.
     */
    public static Complex128[] splice(Field<Complex128>[] arr1, double[] arr2, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(arr1.length + 1, spliceIdx);
        Complex128[] spliced = new Complex128[arr1.length + arr2.length];

        System.arraycopy(arr1, 0, spliced, 0, spliceIdx);
        arraycopy(arr2, 0, spliced, spliceIdx, arr2.length);
        System.arraycopy(arr1, spliceIdx, spliced, spliceIdx + arr2.length, arr1.length - spliceIdx);

        return spliced;
    }


    /**
     * Splices an array into another array at the specified index.
     *
     * @param arr1      First array.
     * @param arr2      Array to splice into {@code arr1}.
     * @param spliceIdx The index within {@code arr1} to splice {@code arr2} into.
     * @return The result of splicing {@code arr2} into {@code arr1} at the index {@code spliceIdx}.
     */
    public static Complex128[] splice(double[] arr1, Field<Complex128>[] arr2, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(arr1.length + 1, spliceIdx);
        Complex128[] spliced = new Complex128[arr1.length + arr2.length];

        arraycopy(arr1, 0, spliced, 0, spliceIdx);
        System.arraycopy(arr2, 0, spliced, spliceIdx, arr2.length);
        arraycopy(arr1, spliceIdx, spliced, spliceIdx + arr2.length, arr1.length - spliceIdx);

        return spliced;
    }


    /**
     * Splices an array into another array at the specified index.
     *
     * @param arr1      First array.
     * @param arr2      Array to splice into {@code arr1}.
     * @param spliceIdx The index within {@code arr1} to splice {@code arr2} into.
     * @return The result of splicing {@code arr2} into {@code arr1} at the index {@code spliceIdx}.
     */
    public static double[] splice(double[] arr1, double[] arr2, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(arr1.length + 1, spliceIdx);
        double[] spliced = new double[arr1.length + arr2.length];

        System.arraycopy(arr1, 0, spliced, 0, spliceIdx);
        System.arraycopy(arr2, 0, spliced, spliceIdx, arr2.length);
        System.arraycopy(arr1, spliceIdx, spliced, spliceIdx + arr2.length, arr1.length - spliceIdx);

        return spliced;
    }


    /**
     * Splices an array into another array at the specified index.
     *
     * @param arr1      First array.
     * @param arr2      Array to splice into {@code arr1}.
     * @param spliceIdx The index within {@code arr1} to splice {@code arr2} into.
     * @return The result of splicing {@code arr2} into {@code arr1} at the index {@code spliceIdx}.
     */
    public static int[] splice(int[] arr1, int[] arr2, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(arr1.length + 1, spliceIdx);
        int[] spliced = new int[arr1.length + arr2.length];

        System.arraycopy(arr1, 0, spliced, 0, spliceIdx);
        System.arraycopy(arr2, 0, spliced, spliceIdx, arr2.length);
        System.arraycopy(arr1, spliceIdx, spliced, spliceIdx + arr2.length, arr1.length - spliceIdx);

        return spliced;
    }


    /**
     * Splices an array into a list at the specified index. The list is implicitly converted to an array.
     *
     * @param list      List to splice into.
     * @param arr       Array to splice into {@code list}.
     * @param spliceIdx The index within {@code list} to splice {@code arr} into.
     * @return The result of splicing {@code arr} into {@code list} at the index {@code spliceIdx}.
     */
    public static double[] splice(List<Double> list, double[] arr, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(list.size() + 1, spliceIdx);
        double[] spliced = new double[list.size() + arr.length];

        for(int i = 0; i < spliceIdx; i++)
            spliced[i] = list.get(i);

        System.arraycopy(arr, 0, spliced, spliceIdx, arr.length);

        for(int i = spliceIdx; i < list.size(); i++)
            spliced[i + arr.length] = list.get(i);

        return spliced;
    }


    /**
     * Splices an array into a list at the specified index. The list is implicitly converted to an array.
     *
     * @param list      List to splice into.
     * @param arr       Array to splice into {@code list}.
     * @param spliceIdx The index within {@code list} to splice {@code arr} into.
     * @return The result of splicing {@code arr} into {@code list} at the index {@code spliceIdx}.
     */
    public static Complex128[] spliceDouble(List<Field<Complex128>> list, double[] arr, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(list.size() + 1, spliceIdx);
        Complex128[] spliced = new Complex128[list.size() + arr.length];

        for (int i = 0; i < spliceIdx; i++)
            spliced[i] = (Complex128) list.get(i);

        ArrayUtils.arraycopy(arr, 0, spliced, spliceIdx, arr.length);

        for(int i = spliceIdx; i < list.size(); i++)
            spliced[i + arr.length] = (Complex128) list.get(i);

        return spliced;
    }


    /**
     * Splices an array into a list at the specified index. The list is implicitly converted to an array.
     *
     * @param list      List to splice into.
     * @param arr       Array to splice into {@code list}.
     * @param spliceIdx The index within {@code list} to splice {@code arr} into.
     * @return The result of splicing {@code arr} into {@code list} at the index {@code spliceIdx}.
     */
    public static Complex128[] splice(List<Field<Complex128>> list, Field<Complex128>[] arr, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(list.size() + 1, spliceIdx);
        Complex128[] spliced = new Complex128[list.size() + arr.length];

        for (int i = 0; i < spliceIdx; i++)
            spliced[i] = (Complex128) list.get(i);

        System.arraycopy(arr, 0, spliced, spliceIdx, arr.length);

        for (int i = spliceIdx; i < list.size(); i++)
            spliced[i + arr.length] = (Complex128) list.get(i);

        return spliced;
    }


    /**
     * Splices an array into a list at the specified index. The list is implicitly converted to an array.
     *
     * @param list      List to splice into.
     * @param arr       Array to splice into {@code list}.
     * @param spliceIdx The index within {@code list} to splice {@code arr} into.
     * @return The result of splicing {@code arr} into {@code list} at the index {@code spliceIdx}.
     */
    public static int[] splice(List<Integer> list, int[] arr, int spliceIdx) {
        ValidateParameters.ensureIndexInBounds(list.size() + 1, spliceIdx);
        int[] spliced = new int[list.size() + arr.length];

        for (int i = 0; i < spliceIdx; i++)
            spliced[i] = list.get(i);

        System.arraycopy(arr, 0, spliced, spliceIdx, arr.length);

        for (int i = spliceIdx; i < list.size(); i++)
            spliced[i + arr.length] = list.get(i);

        return spliced;
    }


    /**
     * Applies a transform to an array. This is done in place.
     * @param src Array to apply transform to. Modified.
     * @param opp Operation to use to transform the array.
     * @return A reference to the {@code src} array.
     */
    public static double[] applyTransform(double[] src, UnaryOperator<Double> opp) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = opp.apply(src[i]);

        return src;
    }


    /**
     * Applies a transform to an array. This is done in place.
     * @param src Array to apply transform to. Modified.
     * @param opp Operation to use to transform the array.
     * @return A reference to the {@code src} array.
     */
    public static <T> T[] applyTransform(T[] src, UnaryOperator<T> opp) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = opp.apply(src[i]);

        return src;
    }


    /**
     * Applies a transform to an array. This is done in place.
     * @param src Array to apply transform to. Modified.
     * @param opp Operation to use to transform the array.
     * @return A reference to the {@code src} array.
     */
    public static <T extends Field<T>> Field<T>[] applyTransform(Field<T>[] src, UnaryOperator<T> opp) {
        for(int i=0, size=src.length; i<size; i++)
            src[i] = opp.apply((T) src[i]);

        return src;
    }
    

    /**
     * Applies a transform to an array. Note, unlike {@link #applyTransform(double[], UnaryOperator)} and
     * {@link #applyTransform(Object[], UnaryOperator)}, this method does <b>not</b> work in place.
     *
     * @param src Array to apply transform to.
     * @param opp Operation to use to transform the array.
     * @return A new array containing the result of the transformation.
     *
     * @see #applyTransform(Object[], UnaryOperator)
     * @see #applyTransform(double[], UnaryOperator)
     */
    public static Complex128[] applyTransform(double[] src, Function<Double, Complex128> opp) {
        Complex128[] dest = new Complex128[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = (Complex128) opp.apply(src[i]);

        return dest;
    }


    /**
     * Applies a transform to an array. Note, unlike {@link #applyTransform(double[], UnaryOperator)} and
     * {@link #applyTransform(Object[], UnaryOperator)}, this method does <b>not</b> work in place.
     *
     * @param src Array to apply transform to.
     * @param opp Operation to use to transform the array.
     * @return A new array containing the result of the transformation.
     *
     * @see #applyTransform(Object[], UnaryOperator)
     * @see #applyTransform(double[], UnaryOperator)
     */
    public static double[] applyTransform(Field<Complex128>[] src, Function<Complex128, Double> opp) {
        double[] dest = new double[src.length];

        for(int i=0, size=src.length; i<size; i++)
            dest[i] = opp.apply((Complex128) src[i]);

        return dest;
    }
}
