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
}
