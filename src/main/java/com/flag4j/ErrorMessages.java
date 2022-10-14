package com.flag4j;

abstract class ErrorMessages {
    private final static String EQ_SHAPE_MISMATCH_ERR = "Expecting matrices to have the same shape but got shapes ";
    private final static String MAT_MULT_DIM_MISMATCH_ERR = "Expecting the number of columns in the first matrix to match the number rows in the" +
            "second matrix but got shapes ";
    private final static String VEC_ROW_ORIENTATION_ERR = "Expecting vector to be a row vector but got a column vector.";
    private final static String VEC_COL_ORIENTATION_ERR = "Expecting vector to be a column vector but got a row vector.";

    /**
     * Gets an error message for two tensors with mismatching shapes.
     * @param shape1 The shape of the first tensor.
     * @param shape2 The shape of the second tensor.
     * @return An error message for tensors with mismatching shapes.
     */
    static String equalShapeErrMsg(Shape shape1, Shape shape2) {
        return EQ_SHAPE_MISMATCH_ERR + shape1 + " and " + shape2 + ".";
    }


    /**
     * Gets an error message for two matrices with shapes not conducive with matrix multiplication.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape of the second matrix.
     * @return An error message for matrices with shapes not conducive with matrix multiplication.
     */
    static String matMultShapeErrMsg(Shape shape1, Shape shape2) {
        return MAT_MULT_DIM_MISMATCH_ERR + shape1 + " and " + shape2 + ".";
    }


    static String vecRowOrientErrMsg() {
        return VEC_ROW_ORIENTATION_ERR;
    }

    static String vecColOrientErrMsg() {
        return VEC_COL_ORIENTATION_ERR;
    }
}

