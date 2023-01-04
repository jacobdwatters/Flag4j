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

package com.flag4j.operations.dense.complex;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains low-level implementations of setting operations for complex dense tensors.
 */
public class ComplexDenseSetOperations {

    private ComplexDenseSetOperations() {
        // Hide constructor.
        throw new IllegalArgumentException(ErrorMessages.utilityClassErrMsg());
    }

    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(CNumber[] src, final CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);
        System.arraycopy(src, 0, dest, 0, src.length);
    }



    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(Double[] src, final CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(Integer[] src, final CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(double[] src, final CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(int[] src, final CNumber[] dest) {
        ParameterChecks.assertArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++) {
            dest[i] = new CNumber(src[i]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(CNumber[][] src, final CNumber[] dest) {
        ParameterChecks.assertTotalEntriesEq(src, dest);
        int count = 0;

        for(CNumber[] cNumbers : src) {
            for(int j = 0; j < src[0].length; j++) {
                dest[count++] = cNumbers[j].clone();
            }
        }
    }



    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(Double[][] src, final CNumber[] dest) {
        ParameterChecks.assertTotalEntriesEq(src, dest);
        int count = 0;

        for(Double[] doubles : src) {
            for(int j = 0; j < src[0].length; j++) {
                dest[count++] = new CNumber(doubles[j]);
            }
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(Integer[][] src, final CNumber[] dest) {
        ParameterChecks.assertTotalEntriesEq(src, dest);
        int count = 0;

        for(Integer[] integers : src) {
            for(int j = 0; j < src[0].length; j++) {
                dest[count++] = new CNumber(integers[j]);
            }
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(double[][] src, final CNumber[] dest) {
        ParameterChecks.assertTotalEntriesEq(src, dest);
        int count = 0;

        for(double[] doubles : src) {
            for(int j = 0; j < src[0].length; j++) {
                dest[count++] = new CNumber(doubles[j]);
            }
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays have different number of total entries.
     */
    public static void setValues(int[][] src, final CNumber[] dest) {
        ParameterChecks.assertTotalEntriesEq(src, dest);
        int count = 0;

        for(int[] ints : src) {
            for(int j = 0; j < src[0].length; j++) {
                dest[count++] = new CNumber(ints[j]);
            }
        }
    }
}
