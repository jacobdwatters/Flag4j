package com.flag4j.util;

import com.flag4j.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ShapeArrayChecksTests {
    Shape shape1, shape2;

    @Test
    void matMultTest() {
        // ------------ Sub-case 1 ------------
        shape1 = new Shape(5, 5);
        shape2 = new Shape(5, 5);

        ShapeArrayChecks.matMultShapeCheck(shape1, shape2);

        // ------------ Sub-case 2 ------------
        shape1 = new Shape(5, 1);
        shape2 = new Shape(1, 5);

        ShapeArrayChecks.matMultShapeCheck(shape1, shape2);

        // ------------ Sub-case 3 ------------
        shape1 = new Shape(1, 5);
        shape2 = new Shape(1, 5);

        assertThrows(IllegalArgumentException.class,
                ()->ShapeArrayChecks.matMultShapeCheck(shape1, shape2));

        // ------------ Sub-case 4 ------------
        shape1 = new Shape(6, 114);
        shape2 = new Shape(114, 6);

        ShapeArrayChecks.matMultShapeCheck(shape1, shape2);

        // ------------ Sub-case 5 ------------
        shape1 = new Shape(112, 1, 1);
        shape2 = new Shape(113);
        assertThrows(IllegalArgumentException.class,
                ()->ShapeArrayChecks.matMultShapeCheck(shape1, shape2));
    }
}
