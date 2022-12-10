package com.flag4j.util;

import com.flag4j.Shape;

public final class ShapeChecks {

    // Hide constructor
    private ShapeChecks() {
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
}
