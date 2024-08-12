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

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.math.BigInteger;
import java.util.Arrays;

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
     * @throws LinearAlgebraException If shapes are not equivalent.
     */
    public static void assertEqualShape(Shape shape1, Shape shape2) {
        if(!shape1.equals(shape2)) {
            throw new LinearAlgebraException(
                    ErrorMessages.equalShapeErrMsg(shape1, shape2)
            );
        }
    }


    /**
     * Checks if two {@link Shape} objects satisfy the requirements of matrix multiplication.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws LinearAlgebraException If shapes do not satisfy the requirements of matrix multiplication.
     */
    public static void assertMatMultShapes(Shape shape1, Shape shape2) {
        if(shape1.getRank() != 2
                || shape2.getRank() != 2
                || shape1.get(1) != shape2.get(0)) {
            throw new LinearAlgebraException(
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
     * Checks if a set of values are all equal.
     * @param values Values to check if they are equal.
     * @throws IllegalArgumentException If any of the specified values are not equal.
     */
    public static void assertEquals(double... values) {
        if(values.length > 0) {
            boolean equal = true;
            double base = values[0];

            for(double v : values) {
                if(v != base) {
                    equal = false;
                    break;
                }
            }

            if(!equal) {
                throw new IllegalArgumentException("Expecting values to be equal but got: " + Arrays.toString(values));
            }
        }
    }


    /**
     * Checks if a set of values are all equal.
     * @param values Values to check if they are equal.
     * @throws IllegalArgumentException If any of the specified values are not equal.
     */
    public static void assertEquals(int... values) {
        if(values.length > 0) {
            boolean equal = true;
            double base = values[0];

            for(double v : values) {
                if(v != base) {
                    equal = false;
                    break;
                }
            }

            if(!equal) {
                throw new IllegalArgumentException("Expecting values to be equal but got: " + Arrays.toString(values));
            }
        }
    }


    /**
     * Checks that two values are not equal.
     * @param a First value.
     * @param b Second value.
     * @throws IllegalArgumentException If {@code a==b}.
     */
    public static void assertNotEquals(double a, double b) {
        if(a==b) {
            throw new IllegalArgumentException("Expecting values to not be equal but got: " + a + ", " + b + ".");
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
     * Checks if a set of values is greater than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If any of the values are less than the threshold.
     */
    public static void assertGreaterEq(int threshold, int... values) {
        for(double value : values) {
            if(value<threshold) {
                throw new IllegalArgumentException(ErrorMessages.getGreaterEqErr(threshold, value));
            }
        }
    }


    /**
     * Checks if a single value is greater than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @throws IllegalArgumentException If the values is less than the threshold.
     */
    public static void assertGreaterEq(int threshold, int value) {
        if(value<threshold) {
            throw new IllegalArgumentException(ErrorMessages.getGreaterEqErr(threshold, value));
        }
    }


    /**
     * Checks if a set of values is greater than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param value Values to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If any of the values are less than the threshold.
     */
    public static void assertGreaterEq(double threshold, double value, String name) {
        if(value<threshold) {
            throw new IllegalArgumentException(ErrorMessages.getNamedGreaterEqErr(threshold, value, name));
        }
    }


    /**
     * Checks if a set of values is less than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If any of the values are greater than the threshold.
     */
    public static void assertLessEq(double threshold, double... values) {
        for(double value : values) {
            if(value>threshold) {
                throw new IllegalArgumentException(ErrorMessages.getLessEqErr(threshold, value));
            }
        }
    }


    /**
     * Checks if a set of values is less than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If any of the values are greater than the threshold.
     */
    public static void assertLessEq(int threshold, int... values) {
        for(double value : values) {
            if(value>threshold) {
                throw new IllegalArgumentException(ErrorMessages.getLessEqErr(threshold, value));
            }
        }
    }


    /**
     * Checks if a value is less than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If the value is greater than the threshold.
     */
    public static void assertLessEq(double threshold, double value, String name) {
        if(value>threshold) throw new IllegalArgumentException(ErrorMessages.getNamedLessEqErr(threshold, value, name));
    }


    /**
     * Checks if a value is less than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If the value is greater than the threshold.
     */
    public static void assertLessEq(BigInteger threshold, int value, String name) {
        if(threshold.compareTo(BigInteger.valueOf(value)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.getNamedLessEqErr(threshold, value, name));
        }
    }


    /**
     * Checks if a value is positive.
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value} is not positive.
     * @see #assertNonNegative(int)
     */
    public static void assertPositive(int value) {
        if(value <= 0) throw new IllegalArgumentException(ErrorMessages.getNonPosErr(value));
    }


    /**
     * Checks if a value is non-negative. Note, this method differs from {@link #assertPositive(int)} as it
     * allows zero values where {@link #assertPositive(int)} does not.
     * value
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value} is negative.
     * @see #assertPositive(int) 
     */
    public static void assertNonNegative(int value) {
        if(value < 0) throw new IllegalArgumentException(ErrorMessages.getNegValueErr(value));
    }


    /**
     * Checks if values are all non-negative according to {@link #assertNonNegative(int)}.
     * @param values Values of interest.
     * @throws IllegalArgumentException If any element of {@code values} is negative.
     * @see #assertPositive(int)
     */
    public static void assertNonNegative(int... values) {
        for(int value : values) {
            assertNonNegative();
        }
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param shape Shape to check.
     * @throws LinearAlgebraException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void assertSquareMatrix(Shape shape) {
        if(shape.getRank()!=2 || shape.get(0)!=shape.get(1)) {
            throw new LinearAlgebraException(ErrorMessages.getSquareShapeErr(shape));
        }
    }


    /**
     * Checks if a shape represents a square tensor.
     * @param shape Shape to check.
     * @throws IllegalArgumentException If all axis of the shape are not the same length.
     */
    public static void assertSquare(Shape shape) {
        ParameterChecks.assertEquals(shape.getDims());
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @throws LinearAlgebraException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void assertSquareMatrix(int numRows, int numCols) {
        if(numRows!=numCols) {
            throw new LinearAlgebraException(ErrorMessages.getSquareShapeErr(new Shape(numRows, numCols)));
        }
    }


    /**
     * Checks that a shape has the specified rank.
     * @param expRank Expected rank.
     * @param shape Shape to check.
     * @throws LinearAlgebraException If the specified shape does not have the expected rank.
     */
    public static void assertRank(int expRank, Shape shape) {
        if(shape.getRank() != expRank) {
            throw new LinearAlgebraException(ErrorMessages.shapeRankErr(shape.getRank(), expRank));
        }
    }


    /**
     * Checks that an axis is a valid 2D axis. That is, either axis is 0 or 1.
     * @param axis Axis to check.
     * @throws IllegalArgumentException If the axis is not a valid 2D axis.
     */
    public static void assertAxis2D(int axis) {
        if(!(axis == 0 || axis==1)) {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, Axis2D.allAxes()));
        }
    }


    /**
     * Checks that a list of {@code N} axis are a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @param axes List of axes of interest.
     * @throws IllegalArgumentException If {@code axis} is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void assertPermutation(int... axes) {
        int[] axesCopy = axes.clone();

        Arrays.sort(axesCopy);

        for(int i=0; i<axesCopy.length; i++) {
            if(axesCopy[i]!=i) {
                throw new IllegalArgumentException("Array is not a permutation of integers 0 through " + axes.length + ".\n" +
                        "Got " + Arrays.toString(axes));
            }
        }
    }


    /**
     * Checks that a value is within the specified inclusive range.
     * @param value Value of interest.
     * @param lowerBound Lower bound of range (inclusive).
     * @param upperBound Upper bound of range (inclusive).
     * @param paramName Name of the parameter.
     * @throws IllegalArgumentException If {@code value} is not within the range {@code [lowerBound, upperBound]}
     */
    public static void assertInRange(double value, double lowerBound, double upperBound, String paramName) {
        if(value < lowerBound || value > upperBound) {
            String name = paramName==null ? "Value" : paramName + " = ";
            String errMsg = String.format("%s %f not in range [%f, %f]", name, value, lowerBound, upperBound);

            throw new IllegalArgumentException(errMsg);
        }
    }


    /**
     * Checks that a set of indices is within {@code [0, upperBound)}.
     * @param upperBound Upper bound of range for indices (exclusive).
     * @param indices Array if indices to check.
     * @throws IndexOutOfBoundsException If any {@code indices} or not within {@code [0, upperBound)}.
     */
    public static void assertIndexInBounds(int upperBound, int... indices) {
        for(int i : indices) {
            if(i < 0 || i >= upperBound) {
                String errMsg = i<0 ?
                        "Index " + i + " is out of bounds for lower bound of 0" :
                        "Index " + i + " is out of bounds for upper bound of " + upperBound + ".";

                throw new IndexOutOfBoundsException(errMsg);
            }
        }
    }


    /**
     * Checks if the provided indices are contained in a tensor defined by the given {@code shape}.
     * @param shape Shape of the tensor.
     * @param indices Indices to check. Must be same length as the number of dimensions in {@code shape}.
     * @throws IndexOutOfBoundsException If {@code indices} is not a valid index into a tensor
     * of the specified {@code shape}.
     */
    public static void assertValidIndex(Shape shape, int... indices) {
        if(shape.getRank() != indices.length) {
            throw new IndexOutOfBoundsException("Expected " + shape.getRank()
                    + " indices but got " + indices.length + ".");
        }

        for(int i=0; i<indices.length; i++) {
            if(indices[i] < 0 || indices[i] >= shape.get(i)) {
                String errMsg = indices[i]<0 ?
                        "Index " + i + " is out of bounds for lower bound of 0" :
                        "Index " + i + " is out of bounds for upper bound of " + shape.get(i) + ".";

                throw new IndexOutOfBoundsException(errMsg);
            }
        }
    }


    /**
     * Checks if the provided indices are contained in an iterable with the given {@code length}.
     * @param length length of iterable.
     * @param indices Indices to check.
     * @throws IndexOutOfBoundsException If {@code indices} is not a valid index into an iterable
     * of the specified {@code length}.
     */
    public static void assertValidIndices(int length, int... indices) {

        for(int i=0; i<indices.length; i++) {
            if(indices[i] < 0 || indices[i] >= length) {
                String errMsg = indices[i]<0 ?
                        "Index " + i + " is out of bounds for lower bound of 0" :
                        "Index " + i + " is out of bounds for upper bound of " + length + ".";

                throw new IndexOutOfBoundsException(errMsg);
            }
        }
    }
}
