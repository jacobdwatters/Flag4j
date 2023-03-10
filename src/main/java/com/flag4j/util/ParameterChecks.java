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

import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;

/**
 * This class contains several methods for checking properties of shapes and arrays.
 */
public final class ParameterChecks {

    // Hide constructor
    private ParameterChecks() {
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if two {@link Shape} objects are equivalent.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws IllegalArgumentException If shapes are not equivalent.
     */
    public static void assertEqualShape(Shape shape1, Shape shape2) {
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
    public static void assertMatMultShapes(Shape shape1, Shape shape2) {
        boolean pass = true;

        // If the shapes are not of rank 2 then they are not matrices.
        if(shape1.getRank()==2 && shape2.getRank()==2) {
            // Ensure the number of columns in matrix one is equal to the number of rows in matrix 2.
            if(shape1.dims[Axis2D.col()] != shape2.dims[Axis2D.row()]) {
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
    public static void assertArrayLengthsEq(int... lengths) {
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
    public static void assertBroadcastable(Shape shape1, Shape shape2) {
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
    public static void assertTotalEntriesEq(Object[][] arr1, double[] arr2) {
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
    public static void assertTotalEntriesEq(double[][] arr1, double[] arr2) {
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
    public static void assertTotalEntriesEq(int[][] arr1, double[] arr2) {
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
    public static void assertTotalEntriesEq(Object[][] arr1, CNumber[] arr2) {
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
    public static void assertTotalEntriesEq(double[][] arr1, CNumber[] arr2) {
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
    public static void assertTotalEntriesEq(int[][] arr1, CNumber[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if a set of values is greater than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If any of the values are less than the threshold.
     */
    public static void assertGreaterEq(double threshold, double... values) {
        for(double value : values) {
            if(value<threshold) {
                throw new IllegalArgumentException(ErrorMessages.getGreaterEqErr(threshold, value));
            }
        }
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param shape Shape to check.
     * @throws IllegalArgumentException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void assertSquare(Shape shape) {
        if(shape.getRank()!=2 || shape.get(0)!=shape.get(1)) {
            throw new IllegalArgumentException(ErrorMessages.getSquareShapeErr(shape));
        }
    }


    /**
     * Checks that a shape has the specified rank.
     * @param expRank Expected rank.
     * @param shape Shape to check.
     * @throws IllegalArgumentException If the specified shape does not have the expected rank.
     */
    public static void assertRank(int expRank, Shape shape) {
        if(shape.getRank() != expRank) {
            throw new IllegalArgumentException(ErrorMessages.shapeRankErr(shape.getRank(), expRank));
        }
    }


    /**
     * Checks that an axis is a valid 2D axis. That is, either axis is 0 or 1.
     * @param axis Axis to check.
     */
    public static void assertAxis2D(int axis) {
        if(!(axis == 0 || axis==1)) {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, Axis2D.allAxes()));
        }
    }
}
