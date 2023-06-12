package com.flag4j.tensor;

import com.flag4j.Shape;
import com.flag4j.Tensor;
import com.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorToStringTests {

    static Shape aShape;
    static double[] aEntries;
    static Tensor A;
    static String exp;

    @BeforeAll
    static void setup() {
        aShape = new Shape(2, 3, 1, 2);
        aEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        A = new Tensor(aShape, aEntries);
    }


    @AfterAll
    static void cleanup() {
        PrintOptions.resetAll();
    }


    @Test
    void toStringTestCase() {
        PrintOptions.resetAll();
        // ---------------------- Sub-case 1 ----------------------
        exp = "Full Shape: 2x3x1x2\n" +
                "[ 1  -1.4133  113.4  0.4  11.3  445  133.445  9.8  13384  ...  12 ]";
        assertEquals(exp, A.toString());

        // ---------------------- Sub-case 2 ----------------------
        PrintOptions.setMaxColumns(15);
        PrintOptions.setPrecision(2);
        PrintOptions.setCentering(false);
        exp = "Full Shape: 2x3x1x2\n" +
                "[1  -1.41  113.4  0.4  11.3  445  133.45  9.8  13384  -993.44  11  12  ]";
        assertEquals(exp, A.toString());
    }
}
