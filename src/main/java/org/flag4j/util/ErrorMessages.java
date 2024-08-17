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


import org.flag4j.core.Shape;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * Contains error messages for common errors which may occur.
 */
public abstract class ErrorMessages {

    /**
     * Hide default constructor.
     */
    private ErrorMessages() {
        throw new IllegalStateException(UTILITY_CLASS_ERR);
    }

    /**
     * Error message for matrices which do not have equal shape.
     */
    private static final String EQ_SHAPE_MISMATCH_ERR = "Expecting matrices to have the same shape but got shapes %s and %s.";
    /**
     * Error message for matrices which do not have valid dimensions for matrix multiplication.
     */
    private static final String MAT_MULT_DIM_MISMATCH_ERR = "Expecting the number of columns in the first matrix to" +
            " match the number rows in the second matrix but got shapes %s and %s.";
    /**
     * Error message for vector which was expected to be row vector.
     */
    private static final String VEC_ROW_ORIENTATION_ERR = "Expecting vector to be a row vector but got a vector with shape %s.";
    /**
     * Error message for vector which was expected to be a column vector.
     */
    private static final String VEC_COL_ORIENTATION_ERR = "Expecting vector to be a column vector but got a row vector with shape %s.";
    /**
     * Error message for the attempted construction of a tensor with a negative dimension value.
     */
    private static final String NEG_DIM_ERR = "Shape dimensions must be non-negative but got shape %s.";
    /**
     * Error message for attempted instantiation of a utility class.
     */
    private static final String UTILITY_CLASS_ERR = "Utility class cannot be instantiated";
    /**
     * Error message for a negative value when a non-negative was expected.
     */
    private static final String NEG_VALUE_ERR = "Expecting value to be non-negative but got %s";
    /**
     * Error message for a non-positive value when a positive non-zero value is expected.
     */
    private static final String NON_POS_ERR = "Expecting value to be positive but got %s";
    /**
     * Error message for disallowed axis.
     */
    private static final String AXIS_ERR_RANGE = "Got an axis of %s but was expecting one of %s";
    /**
     * Error message for disallowed axis.
     */
    private static final String AXIS_ERR = "The axis %s is unspecified.";
    /**
     * Error message wrong shape.
     */
    private static final String SHAPE_RANK_ERR = "Got a shape of rank %d but was expecting a shape of rank %d";
    /**
     * Error message for a shape size which cannot contain a specified number of entries.
     */
    private static final String SHAPE_ENTRIES_ERR = "The shape %s cannot hold %d entries.";
    /**
     * Error message for arrays_old which were expected to be the same length.
     */
    private static final String ARRAY_LENGTHS_MISMATCH_ERR = "Arrays lengths must match but got lengths: %s.";
    /**
     * Error message for a number of indices which does not match the rank of the tensor being indexed.
     */
    private static final String INDICES_RANK_ERR = "Number of indices does not match the rank of the tensor." +
            " Got %s indices but expected %s";
    /**
     * Error message for shapes which cannot be broadcast.
     */
    private static final String SHAPE_BROADCAST_ERR = "Shapes %s and %s can not be broadcast because they specify " +
            "a different number of total entries.";
    /**
     * Error message for arrays_old which do not have the same total number of entries.
     */
    private static final String TOTAL_ENTRIES_ERR = "Arrays do not have the same total number of entries.";
    /**
     * Error message for parameters which are expected to be greater than or equal to some threshold.
     */
    private static final String GREATER_EQ_ERR = "Expecting parameters to be greater than or equal to %f but got %f.";
    /**
     * Error message for names parameter which is expected to be greater than or equal to some threshold.
     */
    private static final String NAMED_GREATER_EQ_ERR = "Expecting %s to be greater than or equal to %f but got %f.";
    /**
     * Error message for parameters which are expected to be less than or equal to some threshold.
     */
    private static final String LESS_EQ_ERR = "Expecting parameters to be less than or equal to %f but got %f.";
    /**
     * Error message for named parameter which is expected to be less than or equal to some threshold.
     */
    private static final String NAMED_LESS_EQ_ERR = "Expecting %s to be less than or equal to %f but got %f.";
    /**
     * Error message for named parameter which is expected to be less than or equal to some threshold.
     */
    private static final String NAMED_LESS_EQ_BI_ERR = "Expecting %s to be less than or equal to %s but got %f.";
    /**
     * Error message for parameters which are expected to be square.
     */
    private static final String SQUARE_SHAPE_ERR = "Expecting matrix to be square but got shape %s";
    /**
     * Error message for a parameter which is not within the expected range.
     */
    private static final String RANGE_ERR = "Expecting parameter %s to be in the range [%f, %f] but got %f";


    /**
     * Gets an error message for two tensors with mismatching shapes.
     * @param shape1 The shape of the first tensor.
     * @param shape2 The shape of the second tensor.
     * @return An error message for tensors with mismatching shapes.
     */
    public static String equalShapeErrMsg(Shape shape1, Shape shape2) {
        return String.format(EQ_SHAPE_MISMATCH_ERR, shape1, shape2);
    }


    /**
     * Gets an error message for two matrices with shapes not conducive with matrix multiplication.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape of the second matrix.
     * @return An error message for matrices with shapes not conducive with matrix multiplication.
     */
    public static String matMultShapeErrMsg(Shape shape1, Shape shape2) {
        return String.format(MAT_MULT_DIM_MISMATCH_ERR, shape1, shape2);
    }


    /**
     * Gets an error message for a vector which was expected to be a row vector but was a column vector.
     * @param shape Shape of the vector.
     * @return An error message for a vector which is not a row vector.
     */
    public static String vecRowOrientErrMsg(Shape shape) {
        return String.format(VEC_ROW_ORIENTATION_ERR, shape);
    }


    /**
     * Gets an error message for a vector which was expected to be a column vector but was a row vector.
     * @param shape Shape of the vector.
     * @return An error message for a vector which is not a row vector.
     */
    public static String vecColOrientErrMsg(Shape shape) {
        return String.format(VEC_COL_ORIENTATION_ERR, shape);
    }


    /**
     * Gets an error message for an attempted construction of a tensor with a negative dimension.
     * @param dims Shape of the tensor.
     * @return An error message for negative dimension.
     */
    public static String negativeDimErrMsg(int[] dims) {
        return String.format(NEG_DIM_ERR,
                Arrays.toString(dims));
    }


    /**
     * Gets an error message for an attempted instantiation of a utility class.
     * @return An error message for the attempted instantiation of a utility class;
     */
    public static String getUtilityClassErrMsg() {return UTILITY_CLASS_ERR;}


    /**
     * Gets an error message for a negative value when a non-negative was expected.
     * @param value Negative value.
     * @return An error message for a negative value.
     */
    public static String getNegValueErr(double value) {
        return String.format(NEG_VALUE_ERR, value);
    }


    /**
     * Gets an error message for a non-positive value when a positive, non-zero value was expected.
     * @param value Non-positive value.
     * @return An error message for a Non-positive value.
     */
    public static String getNonPosErr(double value) {
        return String.format(NON_POS_ERR, value);
    }



    /**
     * Gets an error message for a disallowed axis.
     * @param axis Negative value.
     * @param allowedAxes An array containing allowed axes.
     * @return An error message for a disallowed axis.
     */
    public static String getAxisErr(int axis, int... allowedAxes) {
        return String.format(AXIS_ERR_RANGE, axis, Arrays.toString(allowedAxes));
    }


    /**
     * Gets an error message for a disallowed axis.
     * @param axis Negative value.
     * @return An error message for a disallowed axis.
     */
    public static String getAxisErr(int axis) {
        return String.format(AXIS_ERR, axis);
    }


    /**
     * Gets an error message for an incorrect shape.
     * @param expRank Expected rank of the shape.
     * @param actRank Actual rank of the shape.
     * @return An error message for a shape object with wrong shape.
     */
    public static String shapeRankErr(int actRank, int expRank) {
        return String.format(SHAPE_RANK_ERR, actRank, expRank);
    }


    /**
     * Gets an error message for a shape which cannot hold a specified number of entries.
     * @param shape Shape.
     * @param numEntries Number of entries to hold.
     * @return An error message for a shape which cannot hold a specified number of entries.
     */
    public static String shapeEntriesError(Shape shape, int numEntries) {
        return String.format(SHAPE_ENTRIES_ERR, shape.toString(), numEntries);
    }


    /**
     * Gets an error message for arrays_old which were expected to be the same length.
     * @param lengths Lengths of arrays_old.
     * @return An error message for arrays_old which were expected to be the same length.
     */
    public static String getArrayLengthsMismatchErr(int... lengths) {
        return String.format(ARRAY_LENGTHS_MISMATCH_ERR, Arrays.toString(lengths));
    }


    /**
     * Gets an error message for a tensor being indexed with a number of indices not equal to the rank of the tensor.
     * @param numIndices Number of indices.
     * @param rank Rank of the tensor.
     * @return An error message for a tensor being indexed with a number if indices not equal ot the rank of the tensor.
     */
    public static String getIndicesRankErr(int numIndices, int rank) {
        return String.format(INDICES_RANK_ERR, numIndices, rank);
    }


    /**
     * Gets an error message for two shapes which cannot be broadcast together.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @return An error message for two shapes which cannot be broadcast together.
     */
    public static String getShapeBroadcastErr(Shape shape1, Shape shape2) {
        return String.format(SHAPE_BROADCAST_ERR, shape1, shape2);
    }


    /**
     * Gets an error message for arrays_old which do not have the same total number of entries.
     * @return An error message for arrays_old which do not have the same total number of entries.
     */
    public static String getTotalEntriesErr() {
        return TOTAL_ENTRIES_ERR;
    }


    /**
     * Gets an error message for a value which was expected to be greater than of equal to a specified threshold but wasn't.
     * @param threshold Threshold value.
     * @param value Value expected to be greater than or equal to the threshold value.
     * @return An error message for a value which was expected to be greater than of equal to a
     * specified threshold but wasn't.
     */
    public static String getGreaterEqErr(double threshold, double value) {
        return String.format(GREATER_EQ_ERR, threshold, value);
    }


    /**
     * Gets an error message for a value which was expected to be greater than of equal to a specified threshold but wasn't.
     * @param threshold Threshold value.
     * @param value Value expected to be greater than or equal to the threshold value.
     * @return An error message for a value which was expected to be greater than of equal to a
     * specified threshold but wasn't.
     */
    public static String getNamedGreaterEqErr(double threshold, double value, String name) {
        return String.format(NAMED_GREATER_EQ_ERR, name, threshold, value);
    }

    /**
     * Gets an error message for a value which was expected to be less than of equal to a specified threshold but wasn't.
     * @param threshold Threshold value.
     * @param value Value expected to be less than or equal to the threshold value.
     * @return An error message for a value which was expected to be greater than of equal to a
     * specified threshold but wasn't.
     */
    public static String getLessEqErr(double threshold, double value) {
        return String.format(LESS_EQ_ERR, threshold, value);
    }


    /**
     * Gets an error message for a value which was expected to be less than of equal to a specified threshold but wasn't.
     * @param threshold Threshold value.
     * @param value Value expected to be less than or equal to the threshold value.
     * @return An error message for a value which was expected to be greater than of equal to a
     * specified threshold but wasn't.
     */
    public static String getNamedLessEqErr(double threshold, double value, String name) {
        return String.format(NAMED_LESS_EQ_ERR, name, threshold, value);
    }


    /**
     * Gets an error message for a value which was expected to be less than of equal to a specified threshold but wasn't.
     * @param threshold Threshold value.
     * @param value Value expected to be less than or equal to the threshold value.
     * @return An error message for a value which was expected to be greater than of equal to a
     * specified threshold but wasn't.
     */
    public static String getNamedLessEqErr(BigInteger threshold, double value, String name) {
        return String.format(NAMED_LESS_EQ_BI_ERR, name, threshold, value);
    }


    /**
     * Gets an error message for a shape which was expected to be square but was not
     * @param shape Shape of the tensor.
     * @return An error message for a shape which was expected to be square but was not
     */
    public static String getSquareShapeErr(Shape shape) {
        return String.format(SQUARE_SHAPE_ERR, shape.toString());
    }


    /**
     * Gets an error message for a value which is not within the expected range.
     * @param value Value of interest.
     * @param lowerBound Lower bound of expected range.
     * @param upperBound Upper bound of expected range.
     * @param paramName Name of the parameter which has the specified {@code value}.
     * @return An error message for the parameter being out of the expected range.
     */
    public static String getRangeErr(double value, double lowerBound, double upperBound, String paramName) {
        return String.format(RANGE_ERR, paramName, lowerBound, upperBound, value);
    }

}

