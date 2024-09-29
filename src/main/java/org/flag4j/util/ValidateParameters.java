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
import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * This utility class contains several methods for ensuring a parameter has certain properties.
 */
public final class ValidateParameters {

    // Hide constructor
    private ValidateParameters() {
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if two {@link Shape} objects are equivalent.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws org.flag4j.util.exceptions.TensorShapeException If shapes are not equivalent.
     */
    public static void ensureEqualShape(Shape shape1, Shape shape2) {
        if(!shape1.equals(shape2)) {
            throw new TensorShapeException(
                    ErrorMessages.equalShapeErrMsg(shape1, shape2)
            );
        }
    }


    /**
     * Checks if two {@link Shape} objects satisfy the requirements of matrix multiplication.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws org.flag4j.util.exceptions.TensorShapeException If shapes do not satisfy the requirements of matrix multiplication.
     */
    public static void ensureMatMultShapes(Shape shape1, Shape shape2) {
        if(shape1.getRank() != 2
                || shape2.getRank() != 2
                || shape1.get(1) != shape2.get(0)) {
            throw new TensorShapeException(
                    ErrorMessages.matMultShapeErrMsg(shape1, shape2)
            );
        }
    }


    /**
     * Checks that all array lengths are equal.
     * @param lengths An array of array lengths.
     * @throws IllegalArgumentException If all lengths are not equal.
     */
    public static void ensureArrayLengthsEq(int... lengths) {
        for(int i=0; i<lengths.length-1; i++) {
            if(lengths[i]!=lengths[i+1]) {
                throw new IllegalArgumentException(ErrorMessages.getArrayLengthsMismatchErr(lengths));
            }
        }
    }


    /**
     * Checks that two shapes can be broadcast, i.e. have the same total number of entries.
     * @param shape1 First shape to compare.
     * @param shape2 Second shape to compare.
     * @throws TensorShapeException If the two shapes do not have the same total number of entries.
     */
    public static void ensureBroadcastable(Shape shape1, Shape shape2) {
        if(!shape1.totalEntries().equals(shape2.totalEntries())) {
            throw new TensorShapeException(ErrorMessages.getShapeBroadcastErr(shape1, shape2));
        }
    }


    /**
     * Checks if arrays_old have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays_old do not have the same number of total entries.
     */
    public static void ensureTotalEntriesEq(Object[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays_old have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays_old do not have the same number of total entries.
     */
    public static void ensureTotalEntriesEq(double[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays_old have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays_old do not have the same number of total entries.
     */
    public static void ensureTotalEntriesEq(int[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays_old have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays_old do not have the same number of total entries.
     */
    public static void ensureTotalEntriesEq(Object[][] arr1, Complex128[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays_old have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays_old do not have the same number of total entries.
     */
    public static void ensureTotalEntriesEq(double[][] arr1, Complex128[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if arrays_old have the same number of total entries.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays_old do not have the same number of total entries.
     */
    public static void ensureTotalEntriesEq(int[][] arr1, Complex128[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length) {
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
        }
    }


    /**
     * Checks if a set of values are all equal.
     * @param values Values to check if they are equal.
     * @throws IllegalArgumentException If any of the specified values are not equal.
     */
    public static void ensureEquals(double... values) {
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
    public static void ensureEquals(int... values) {
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
    public static void ensureNotEquals(double a, double b) {
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
    public static void ensureGreaterEq(double threshold, double... values) {
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
    public static void ensureGreaterEq(int threshold, int... values) {
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
    public static void ensureGreaterEq(int threshold, int value) {
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
    public static void ensureGreaterEq(double threshold, double value, String name) {
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
    public static void ensureLessEq(double threshold, double... values) {
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
    public static void ensureLessEq(int threshold, int... values) {
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
    public static void ensureLessEq(double threshold, double value, String name) {
        if(value>threshold) throw new IllegalArgumentException(ErrorMessages.getNamedLessEqErr(threshold, value, name));
    }


    /**
     * Checks if a value is less than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If the value is greater than the threshold.
     */
    public static void ensureLessEq(BigInteger threshold, int value, String name) {
        if(threshold.compareTo(BigInteger.valueOf(value)) < 0) {
            throw new IllegalArgumentException(ErrorMessages.getNamedLessEqErr(threshold, value, name));
        }
    }


    /**
     * Checks if a value is positive.
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value} is not positive.
     * @see #ensureNonNegative(int)
     */
    public static void ensurePositive(int value) {
        if(value <= 0) throw new IllegalArgumentException(ErrorMessages.getNonPosErr(value));
    }


    /**
     * Checks if a value is non-negative. Note, this method differs from {@link #ensurePositive(int)} as it
     * allows zero values where {@link #ensurePositive(int)} does not.
     * value
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value} is negative.
     * @see #ensurePositive(int)
     */
    public static void ensureNonNegative(int value) {
        if(value < 0) throw new IllegalArgumentException(ErrorMessages.getNegValueErr(value));
    }


    /**
     * Checks if values are all non-negative according to {@link #ensureNonNegative(int)}.
     * @param values Values of interest.
     * @throws IllegalArgumentException If any element of {@code values} is negative.
     * @see #ensurePositive(int)
     */
    public static void ensureNonNegative(int... values) {
        for(int value : values)
            ensureNonNegative(value);
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param shape Shape to check.
     * @throws LinearAlgebraException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void ensureSquareMatrix(Shape shape) {
        if(shape.getRank()!=2 || shape.get(0)!=shape.get(1)) {
            throw new LinearAlgebraException(ErrorMessages.getSquareShapeErr(shape));
        }
    }


    /**
     * Checks if a shape represents a square tensor.
     * @param shape Shape to check.
     * @throws IllegalArgumentException If all axis of the shape are not the same length.
     */
    public static void ensureSquare(Shape shape) {
        ValidateParameters.ensureEquals(shape.getDims());
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @throws LinearAlgebraException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void ensureSquareMatrix(int numRows, int numCols) {
        if(numRows!=numCols) {
            throw new LinearAlgebraException(ErrorMessages.getSquareShapeErr(new Shape(numRows, numCols)));
        }
    }


    /**
     * Checks that a shape has the specified rank.
     *
     * @param shape Shape to check.
     * @param expRank Expected rank.
     *
     * @throws LinearAlgebraException If the specified shape does not have the expected rank.
     */
    public static void ensureRank(Shape shape, int expRank) {
        if(shape.getRank() != expRank) {
            throw new LinearAlgebraException(ErrorMessages.shapeRankErr(shape.getRank(), expRank));
        }
    }


    /**
     * Checks that an axis is a valid 2D axis. That is, either axis is 0 or 1.
     * @param axis Axis to check.
     * @throws IllegalArgumentException If the axis is not a valid 2D axis.
     */
    public static void ensureAxis2D(int axis) {
        if(!(axis == 0 || axis==1)) {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, Axis2D.allAxes()));
        }
    }


    /**
     * Checks that a list of {@code N} axis are a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @param axes List of axes of interest.
     * @throws IllegalArgumentException If {@code axis} is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void ensurePermutation(int... axes) {
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
    public static void ensureInRange(double value, double lowerBound, double upperBound, String paramName) {
        if(value < lowerBound || value > upperBound) {
            String name = paramName==null ? "Value" : paramName + " = ";
            String errMsg = String.format("%s %f not in range [%f, %f]", name, value, lowerBound, upperBound);

            throw new IllegalArgumentException(errMsg);
        }
    }


    /**
     * Checks that a set of indices is within {@code [0, upperBound)}.
     * @param upperBound Upper bound of range for indices (exclusive).
     * @param indices Array of indices to check.
     * @throws IndexOutOfBoundsException If any {@code indices} or not within {@code [0, upperBound)}.
     */
    public static void ensureIndexInBounds(int upperBound, int... indices) {
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
    public static void ensureValidIndex(Shape shape, int... indices) {
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
    public static void ensureValidIndices(int length, int... indices) {
        for(int i=0, size=indices.length; i<size; i++) {
            if(indices[i] < 0 || indices[i] >= length) {
                String errMsg = indices[i]<0 ?
                        "Index " + i + " is out of bounds for lower bound of 0" :
                        "Index " + i + " is out of bounds for upper bound of " + length + ".";

                throw new IndexOutOfBoundsException(errMsg);
            }
        }
    }


    /**
     * <p>Checks if all provided {@code axes} are valid with respect to the rank of the given {@code shape}.</p>
     * <p>Specifically, an axis is valid if {@code axis >= 0 && axis < shape.getRank()}.</p>
     * @param shape Shape of interest.
     * @param axes Axes to validate.
     * @throws LinearAlgebraException If {@code axis < 0 || axis >= shape.getRank()} for any axis in {@code axes}.
     */
    public static void ensureValidAxes(Shape shape, int... axes) {
        int rank = shape.getRank();

        for(int axis : axes) {
            if(axis < 0 || axis >= rank) {
                throw new LinearAlgebraException(
                        String.format("Axis %d is out of bounds for shape %s with rank %d.", axis, shape, rank)
                );
            }
        }
    }


    /**
     * Checks if an array's length is equal to the rank of a {@code shape}.
     * @param shape Shape of interest.
     * @param size Size of the array.
     * @throws LinearAlgebraException If {@code size != shape.getShape()}.
     */
    public static void ensureLengthEqualsRank(Shape shape, int size) {
        if(shape.getRank() != size)
            throw new LinearAlgebraException("Array length of " + size + " does not match rank of " + shape.getRank());
    }
}
