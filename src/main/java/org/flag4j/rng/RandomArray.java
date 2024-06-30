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

package org.flag4j.rng;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * This class contains methods useful for generating arrays filled with random values.
 */
public final class RandomArray {

    /**
     * Random number generator to use when creating random arrays.
     */
    private final RandomCNumber rng;


    /**
     * Creates a RandomArray object to generate arrays filled with random values using a default random number generator.
     */
    public RandomArray() {
        this.rng = new RandomCNumber();
    }


    /**
     * Creates a RandomArray object to generate arrays filled with random values using the specified complex
     * random number generator.
     * @param rng The complex random number generator to use when creating random arrays.
     */
    public RandomArray(RandomCNumber rng) {
        this.rng = rng;
    }


    /**
     * Generates an array of doubles filled with uniformly distributed pseudorandom values in {@code [0.0, 1.0)}.
     * To generate uniformly distributed values in a specified range see {@link #genUniformRealArray(int, double, double)}.
     * @param length Length of pseudorandom array to generate.
     * @return An array of doubles with specified length filled with uniformly distributed values in {@code [0.0, 1.0)}.
     */
    public double[] genUniformRealArray(int length) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.nextDouble();
        }

        return values;
    }


    /**
     * Generates an array of doubles filled with uniformly distributed pseudorandom values in {@code [min, max)}.
     * @param length Length of pseudorandom array to generate.
     * @param min Lower bound of uniform range (inclusive).
     * @param max Upper bound of uniform range (Exclusive).
     * @return An array of doubles with specified length filled with uniformly distributed values in {@code [min, max)}.
     */
    public double[] genUniformRealArray(int length, double min, double max) {
        double[] values = new double[length];
        double maxMin = max-min;

        for(int i=0; i<length; i++) {
            values[i] = rng.nextDouble()*maxMin + min;
        }

        return values;
    }


    /**
     * Generates an array of integers filled with uniformly distributed pseudorandom int values in
     * [{@link Integer#MAX_VALUE}, {@link Integer#MAX_VALUE}).
     * To generate uniformly distributed values in a specified range see {@link #genUniformRealIntArray(int, int, int)}.
     * @param length Length of pseudorandom array to generate.
     * @return An array of integers with specified length filled with uniformly distributed values in
     * [{@link Integer#MAX_VALUE}, {@link Integer#MAX_VALUE}).
     */
    public double[] genUniformRealIntArray(int length) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.nextInt();
        }

        return values;
    }


    /**
     * Generates an array of integers filled with uniformly distributed pseudorandom values in {@code [min, max)}.
     * @param length Length of pseudorandom array to generate.
     * @param min Lower bound of uniform range (inclusive).
     * @param max Upper bound of uniform range (Exclusive).
     * @return An array of integers with specified length filled with uniformly distributed values in {@code [min, max)}.
     */
    public int[] genUniformRealIntArray(int length, int min, int max) {
        int[] values = new int[length];
        int maxMinDiff = max - min;

        for(int i=0; i<length; i++) {
            values[i] = rng.nextInt(maxMinDiff) + min;
        }

        return values;
    }


    /**
     * Generates an array of doubles filled with normally pseudorandom values with a mean of 0 and standard deviation
     * of 1.
     * @param length Length of pseudorandom array to generate.
     * @return An array of doubles with specified length filled with normally pseudorandom values with a mean of 0 and
     * standard deviation of 1.
     */
    public double[] genNormalRealArray(int length) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.nextGaussian();
        }

        return values;
    }


    /**
     * Generates an array of doubles filled with normally distributed pseudorandom values with a specified mean and standard deviation.
     * @param length Length of pseudorandom array to generate.
     * @param mean Mean of normal distribution.
     * @param std Standard deviation of normal distribution.
     * @return An array of doubles with specified length filled with normally pseudorandom values specified mean and
     * standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public double[] genNormalRealArray(int length, double mean, double std) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.nextGaussian()*mean + std;
        }

        return values;
    }


    /**
     * Generates an array of {@link CNumber complex numbers} with pseudorandom uniformly distributed magnitudes
     * in {@code [0.0, 1.0)}.
     * @param length Length of the pseudorandom array to generate.
     * @return An array of {@link CNumber complex numbers} with pseudorandom uniformly distributed magnitudes
     * in {@code [0.0, 1.0)}.
     */
    public CNumber[] genUniformComplexArray(int length) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.random();
        }

        return values;
    }


    /**
     * Generates an array of pseudorandom {@link CNumber complex numbers} with uniformly distributed magnitudes
     * in {@code [min, max)}.
     * @param length Length of the pseudorandom array to generate.
     * @param min Minimum value of uniform distribution from which the magnitudes are sampled (inclusive).
     * @param max Maximum value of uniform distribution from which the magnitudes are sampled (exclusive).
     * @return An array of {@link CNumber complex numbers} with pseudorandom {@link CNumber complex numbers} with
     * uniformly distributed magnitudes in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is negative or if {@code max} is less than {@code min}.
     */
    public CNumber[] genUniformComplexArray(int length, double min, double max) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.random(min, max);
        }

        return values;
    }


    /**
     * Generates an array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with a mean of 0.0 and a magnitude of 1.0.
     * @param length Length of the pseudorandom array to generate.
     * @return An array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with a mean of 0.0 and a magnitude of 1.0.
     */
    public CNumber[] genNormalComplexArray(int length) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.randn();
        }

        return values;
    }


    /**
     * Generates an array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with specified mean and standard deviation.
     * @param length Length of the pseudorandom array to generate.
     * @param mean Mean of the normal distribution from which to sample magnitudes.
     * @param std Standard deviation of the normal distribution from which to sample magnitudes.
     * @return An array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public CNumber[] genNormalComplexArray(int length, double mean, double std) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = rng.randn(mean, std);
        }

        return values;
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
        ParameterChecks.assertIndexInBounds(end, start);

        int[] indices = ArrayUtils.intRange(start, end);
        shuffle(indices); // Shuffle indices.

        indices = Arrays.copyOfRange(indices, 0, numIndices); // Extract first 'numIndices' entries.
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
     * @return A two-dimensional array of shape {@code 2-by-numIndices} containing unique two-dimensional indices.
     * The first row contains row indices, the second, column indices. The indices will be sorted by rows then columns.
     * @see #randomUniqueIndices(int, int, int)
     */
    public int[][] randomUniqueIndices2D(int numIndices, int rowStart, int rowEnd, int colStart, int colEnd) {
        ParameterChecks.assertGreaterEq(0, numIndices);
        ParameterChecks.assertLessEq((rowEnd-rowStart)*(colEnd-colStart), numIndices);

        int[] colIndices = new int[numIndices];
//        int[] rowIndices = genUniformRealIntArray(numIndices, rowStart, rowEnd); // Get random row indices.
//        Arrays.sort(rowIndices);
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
    public int[] shuffle(int[] arr) {
        for (int i = arr.length-1; i>0; i--) {

            // Pick a random index from 0 to i
            int j = rng.nextInt(i+1);

            // Swap arr[i] with the element at random index
            ArrayUtils.swap(arr, i, j);
        }

        return arr;
    }


    /**
     * Randomly shuffles array using the Fisher–Yates algorithm. This is done in place.
     *
     * @param arr Array to shuffle.
     */
    public void shuffle(double[] arr) {
        for (int i = arr.length-1; i>0; i--) {

            // Pick a random index from 0 to i
            int j = rng.nextInt(i+1);

            // Swap arr[i] with the element at random index
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

            // Pick a random index from 0 to i
            int j = rng.nextInt(i+1);

            // Swap arr[i] with the element at random index
            ArrayUtils.swap(arr, i, j);
        }
    }
}
