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

package org.flag4j.operations.dense.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains low-level implementations of setting operations for complex dense tensors.
 */
public final class ComplexDenseSetOperations {

    private ComplexDenseSetOperations() {
        // Hide constructor.
        throw new IllegalArgumentException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(Double[] src, Complex128[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++)
            dest[i] = new Complex128(src[i]);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(Integer[] src, Complex128[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++)
            dest[i] = new Complex128(src[i]);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(double[] src, Complex128[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++)
            dest[i] = new Complex128(src[i]);
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(int[] src, Complex128[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src.length, dest.length);

        for(int i=0; i<src.length; i++)
            dest[i] = new Complex128(src[i]);
    }



    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(Double[][] src, Complex128[] dest) {
        ValidateParameters.ensureTotalEntriesEq(src, dest);
        int count = 0;

        for(Double[] doubles : src) {
            for(int j = 0; j < src[0].length; j++)
                dest[count++] = new Complex128(doubles[j]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(Integer[][] src, Complex128[] dest) {
        ValidateParameters.ensureTotalEntriesEq(src, dest);
        int count = 0;

        for(Integer[] integers : src) {
            for(int j = 0; j < src[0].length; j++)
                dest[count++] = new Complex128(integers[j]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(double[][] src, Complex128[] dest) {
        ValidateParameters.ensureTotalEntriesEq(src, dest);
        int count = 0;

        for(double[] doubles : src) {
            for(int j = 0; j < src[0].length; j++)
                dest[count++] = new Complex128(doubles[j]);
        }
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param src New values of the matrix.
     * @param dest Destination array for values.
     * @throws IllegalArgumentException If the source and destination arrays_old have different number of total entries.
     */
    public static void setValues(int[][] src, Complex128[] dest) {
        ValidateParameters.ensureTotalEntriesEq(src, dest);
        int count = 0;

        for(int[] ints : src) {
            for(int j = 0; j < src[0].length; j++)
                dest[count++] = new Complex128(ints[j]);
        }
    }


    /**
     * Sets an element of a tensor to the specified value.
     * @param src Elements of the tensor. This will be modified.
     * @param shape Shape of the tensor.
     * @param value Value to set specified index to.
     * @param indices Indices of tensor value to be set.
     */
    public static void set(Complex128[] src, Shape shape, double value, int... indices) {
        src[shape.entriesIndex(indices)] = new Complex128(value);
    }
}
