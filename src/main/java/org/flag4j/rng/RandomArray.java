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

package org.flag4j.rng;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.algebraic_structures.Complex64;
import org.flag4j.rng.distributions.Distribution;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public final class RandomArray {

    /**
     * Random number generator to use when creating random arrays.
     */
    private final RandomComplex rng;


    /**
     * Creates a RandomArray object to generate arrays filled with random values using a default random number generator.
     */
    public RandomArray() {
        this.rng = new RandomComplex();
    }


    /**
     * Creates a RandomArray object to generate arrays filled with random values using the specified complex
     * random number generator.
     * @param rng The complex random number generator to use when creating random arrays.
     */
    public RandomArray(RandomComplex rng) {
        this.rng = rng;
    }


    /**
     * Fills an array with pseudorandom values sampled from the specified {@code distribution}.
     * @param arr The array to fill.
     * @param distribution The distribution to sample from.
     */
    public static void randomFill(int[] arr, Distribution<Integer, Random> distribution) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = distribution.sample();
    }


    /**
     * Fills an array with pseudorandom values sampled from the specified {@code distribution}.
     * @param arr The array to fill.
     * @param distribution The distribution to sample from.
     */
    public static void randomFill(double[] arr, Distribution<Double, Random> distribution) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = distribution.sample();;
    }


    /**
     * Fills an array with pseudorandom values sampled from the specified {@code distribution}.
     * @param arr The array to fill.
     * @param distribution The distribution to sample from.
     */
    public static void randomFill(Complex128[] arr, Distribution<Complex128, RandomComplex> distribution) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = distribution.sample();
    }


    /**
     * Fills an array with pseudorandom values sampled from the specified {@code distribution}.
     * @param arr The array to fill.
     * @param distribution The distribution to sample from.
     */
    public static void randomFill(Complex64[] arr, Distribution<Complex64, RandomComplex> distribution) {
        for(int i=0, size=arr.length; i<size; i++)
            arr[i] = distribution.sample();
    }


    /**
     * Creates unique indices in [start, end).
     * @param numIndices Number of random unique indices to get.
     * @param start Staring index (inclusive).
     * @param end Ending index (exclusive).
     * @return An array of length {@code numIndices} containing random unique indices in [start, end). The array will be
     * sorted.
     * @see #randomUniqueIndices2D(int, int, int, int, int)
     * @throws IllegalArgumentException If {@code start} is not in {@code [0, end)}
     */
    public int[] randomUniqueIndices(int numIndices, int start, int end) {
        ValidateParameters.validateArrayIndices(end, start);

        int[] indices = ArrayBuilder.intRange(start, end);
        shuffle(indices); // Shuffle indices.

        indices = Arrays.copyOfRange(indices, 0, numIndices); // Extract first 'numIndices' data.
        Arrays.sort(indices); // Sort indices.

        return indices;
    }


    /**
     * Creates a list of unique two-dimensional indices.
     * @param numIndices Total number of indices to generate.
     * @param rowStart Starting row index (inclusive).
     * @param rowEnd Ending row index (exclusive).
     * @param colStart Starting column index (inclusive).
     * @param colEnd Ending column index (exclusive).
     * @return A two-dimensional array of shape {@code 2&times;numIndices} containing unique two-dimensional indices.
     * The first row contains row indices, the second, column indices. The indices will be sorted by rows then columns.
     * @see #randomUniqueIndices(int, int, int)
     */
    public int[][] randomUniqueIndices2D(int numIndices, int rowStart, int rowEnd, int colStart, int colEnd) {
        ValidateParameters.ensureGreaterEq(0, numIndices);
        ValidateParameters.ensureLessEq((rowEnd-rowStart)*(colEnd-colStart), numIndices);

        int[] colIndices = new int[numIndices];
        int[] rowIndices = genRandomRows(numIndices, rowStart, rowEnd, colEnd-colStart);

        // Generate unique column indices for each row index.
        int idx = 0;
        while(idx < numIndices) {
            // Find first and last occurrence of the row index.
            int[] startEnd = ArrayUtils.findFirstLast(rowIndices, rowIndices[idx]);

            // Generate unique row indices for the specified row and copy into .
            int[] uniqueCols = randomUniqueIndices(startEnd[1]-startEnd[0], colStart, colEnd);
            System.arraycopy(uniqueCols, 0, colIndices, startEnd[0], uniqueCols.length);

            idx = startEnd[1]; // Update the index.
        }

        return new int[][]{rowIndices, colIndices};
    }


    /**
     * Helper function to generate random row indices for use in {@link #randomUniqueIndices2D(int, int, int, int, int)}.
     * This method generates random row indices so that a single row is not repeated more than {@code maxReps} times.
     * @param numIndices Number of indices to generate. Assumed to be less than or equal to {@code rowEnd - rowStart}.
     * @param rowStart Minimum row index. Assumed that {@code 0 <= rowStart < rowEnd}.
     * @param rowEnd Maximum row index. Assumed that {@code 0 <= rowStart < rowEnd}.
     * @param maxReps Maximum number of times a single row index can be repeated.
     * @return An array containing random row indices such that no index is repeated more than {@code maxReps} times.
     * Note: the array is sorted before it is returned.
     */
    private int[] genRandomRows(int numIndices, int rowStart, int rowEnd, int maxReps) {
        int[] rowIndices = new int[numIndices];
        int maxMinDiff = rowEnd - rowStart;
        int validCount = 0;
        Map<Integer, Integer> map = new HashMap<>(maxMinDiff); // Key=index, value=number of occurrences

        while(validCount < numIndices) {
            int value = rng.nextInt(maxMinDiff) + rowStart; // Generate index in the specified bounds.
            int occurrences = map.getOrDefault(value, 0);

            // Ensure the value has not already been generated more than the maximum number of allowed times.
            if(occurrences < maxReps) {
                map.put(value, occurrences+1);
                rowIndices[validCount++] = value;
            }
        }

        Arrays.sort(rowIndices);
        return rowIndices;
    }


    /**
     * Randomly shuffles array using the Fisher–Yates algorithm. This is done in place.
     *
     * @param arr Array to shuffle.
     * @return A reference to {@code arr}.
     */
    public void shuffle(int[] arr) {
        for (int i = arr.length-1; i>0; i--) {
            // Pick a random index from 0 to i
            int j = rng.nextInt(i+1);
            // Swap arr[i] with the element at random index.
            ArrayUtils.swap(arr, i, j);
        }
    }


    /**
     * Randomly shuffles array using the Fisher–Yates algorithm. This is done in place.
     *
     * @param arr Array to shuffle.
     */
    public void shuffle(double[] arr) {
        for (int i = arr.length-1; i>0; i--) {
            // Pick a random index from 0 to i.
            int j = rng.nextInt(i+1);
            // Swap arr[i] with the element at random index.
            ArrayUtils.swap(arr, i, j);
        }
    }


    /**
     * Randomly shuffles array using the Fisher–Yates algorithm. This is done in place.
     *
     * @param arr Array to shuffle.
     */
    public void shuffle(Object[] arr) {
        for (int i = arr.length-1; i>0; i--) {
            // Pick a random index from 0 to i.
            int j = rng.nextInt(i+1);
            // Swap arr[i] with the element at random index.
            ArrayUtils.swap(arr, i, j);
        }
    }
}
