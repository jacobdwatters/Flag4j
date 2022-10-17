package com.flag4j;


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
    private static final String UTILITY_CLASS_ERR = "Utility class cannot be instantiated";


    /**
     * Gets an error message for two tensors with mismatching shapes.
     * @param shape1 The shape of the first tensor.
     * @param shape2 The shape of the second tensor.
     * @return An error message for tensors with mismatching shapes.
     */
    static String equalShapeErrMsg(Shape shape1, Shape shape2) {
        return String.format(EQ_SHAPE_MISMATCH_ERR, shape1, shape2);
    }


    /**
     * Gets an error message for two matrices with shapes not conducive with matrix multiplication.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape of the second matrix.
     * @return An error message for matrices with shapes not conducive with matrix multiplication.
     */
    static String matMultShapeErrMsg(Shape shape1, Shape shape2) {
        return String.format(MAT_MULT_DIM_MISMATCH_ERR, shape1, shape2);
    }


    /**
     * Gets an error message for a vector which was expected to be a row vector but was a column vector.
     * @param shape Shape of the vector.
     * @return An error message for a vector which is not a row vector.
     */
    static String vecRowOrientErrMsg(Shape shape) {
        return String.format(VEC_ROW_ORIENTATION_ERR, shape);
    }


    /**
     * Gets an error message for a vector which was expected to be a column vector but was a row vector.
     * @param shape Shape of the vector.
     * @return An error message for a vector which is not a row vector.
     */
    static String vecColOrientErrMsg(Shape shape) {
        return String.format(VEC_COL_ORIENTATION_ERR, shape);
    }


    /**
     * Gets an error message for an attempted construction of a tensor with a negative dimension.
     * @param shape Shape of the tensor.
     * @return
     */
    static String negativeDimErrMsg(Shape shape) {
        return String.format(NEG_DIM_ERR, shape);
    }

    public static String utilityClassErrMsg() {return UTILITY_CLASS_ERR;}
}

