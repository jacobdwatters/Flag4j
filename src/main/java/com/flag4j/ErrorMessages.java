package com.flag4j;


/**
 * Contains error messages for common errors which may occur.
 */
abstract class ErrorMessages {

    /**
     * Error message for matrices which do not have equal shape.
     */
    private final static String EQ_SHAPE_MISMATCH_ERR = "Expecting matrices to have the same shape but got shapes %s and %s.";
    /**
     * Error message for matrices which do not have valid dimensions for matrix multiplication.
     */
    private final static String MAT_MULT_DIM_MISMATCH_ERR = "Expecting the number of columns in the first matrix to" +
            " match the number rows in the second matrix but got shapes %s and %s.";
    /**
     * Error message for vector which was expected to be row vector.
     */
    private final static String VEC_ROW_ORIENTATION_ERR = "Expecting vector to be a row vector but got a vector with shape %s.";
    /**
     * Error message for vector which was expected to be a column vector.
     */
    private final static String VEC_COL_ORIENTATION_ERR = "Expecting vector to be a column vector but got a row vector with shape %s.";


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
}

