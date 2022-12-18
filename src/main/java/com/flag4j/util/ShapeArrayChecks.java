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

import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;

/**
 * This class contains several methods for checking properties of shapes and arrays.
 */
public final class ShapeArrayChecks {

    // Hide constructor
    private ShapeArrayChecks() {
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Checks if two {@link Shape} objects are equivalent.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws IllegalArgumentException If shapes are not equivalent.
     */
    public static void equalShapeCheck(Shape shape1, Shape shape2) {
        if(!shape1.equals(shape2)) {
            throw new IllegalArgumentException(
                    ErrorMessages.equalShapeErrMsg(shape1, shape2)
            );
        }
    }


    /**
     * Checks if two {@link Shape} objects satisfy the requirements of matrix multiplication.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws IllegalArgumentException If shapes do not satisfy the requirements of matrix multiplication.
     */
    public static void matMultShapeCheck(Shape shape1, Shape shape2) {
        boolean pass = true;

        // If the shapes are not of rank 2 then they are not matrices.
        if(shape1.getRank()==2 && shape2.getRank()==2) {
            // Ensure the number of columns in matrix one is equal to the number of rows in matrix 2.
            if(shape1.dims[1] != shape2.dims[0]) {
                pass = false;
            }

        } else {
            pass = false;
        }

        if(!pass) { // Check if the shapes pass the test.
            throw new IllegalArgumentException(
                    ErrorMessages.matMultShapeErrMsg(shape1, shape2)
            );
        }
    }


    /**
     * Checks that all array lengths are equal.
     * @param lengths An array of array lengths.
     * @throws IllegalArgumentException If all lengths are not equal.
     */
    public static void arrayLengthsCheck(int... lengths) {
        boolean allEqual = true;

        for(int i=0; i<lengths.length-1; i++) {
            if(lengths[i]!=lengths[i+1]) {
                allEqual=false;
                break;
            }
        }

        if(!allEqual) {
            throw new IllegalArgumentException(ErrorMessages.getArrayLengthsMismatchErr(lengths));
        }
    }


    /**
     * Checks that two shapes can be broadcast, i.e. have the same total number of entries.
     * @param shape1 First shape to compare.
     * @param shape2 Second shape to compare.
     * @throws IllegalArgumentException If the two shapes do not have the same total number of entries.
     */
    public static void broadcastCheck(Shape shape1, Shape shape2) {
        if(!shape1.totalEntries().equals(shape2.totalEntries())) {
            throw new IllegalArgumentException(ErrorMessages.getShapeBroadcastErr(shape1, shape2));
        }
    }


    /**
     * Checks if arrays have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total entries.
     */
    public static void equalTotalEntries(Object[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total entries.
     */
    public static void equalTotalEntries(double[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total entries.
     */
    public static void equalTotalEntries(int[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total entries.
     */
    public static void equalTotalEntries(Object[][] arr1, CNumber[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total entries.
     */
    public static void equalTotalEntries(double[][] arr1, CNumber[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total entries.
     */
    public static void equalTotalEntries(int[][] arr1, CNumber[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }
}
