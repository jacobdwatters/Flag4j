package com.flag4j.util;


import com.flag4j.Shape;

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
     * Error message for disallowed axis.
     */
    private static final String AXIS_ERR = "Got an axis of %s but was expecting one of %s";


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
     * @param shape Shape of the tensor.
     * @return An error message for negative dimension.
     */
    public static String negativeDimErrMsg(Shape shape) {
        return String.format(NEG_DIM_ERR, shape);
    }


    /**
     * Gets an error message for an attempted instantiation of a utility class.
     * @return An error message for the attempted instantiation of a utility class;
     */
    public static String utilityClassErrMsg() {return UTILITY_CLASS_ERR;}


    /**
     * Gets an error message for a negative value when a non-negative was expected.
     * @param value Negative value.
     * @return An error message for a negative value.
     */
    public static String negValueErr(double value) {
        return String.format(NEG_VALUE_ERR, value);
    }


    /**
     * Gets an error message for a disallowed axis.
     * @param axis Negative value.
     * @param allowedAxes An array containing allowed axes.
     * @return An error message for a disallowed axis.
     */
    public static String axisErr(int axis, int[] allowedAxes) {
        return String.format(NEG_VALUE_ERR, axis, Arrays.toString(allowedAxes));
    }
}

