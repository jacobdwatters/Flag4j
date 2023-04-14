package com.flag4j.tensor;

import com.flag4j.Shape;
import com.flag4j.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TensorNormTests {

    static Shape aShape;
    static double[] aEntries;
    static Tensor A;
    static double exp;

    @BeforeAll
    static void setup() {
        aShape = new Shape(2, 3, 1, 2);
        aEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void infNormTestCase() {
        // ------------------------- Sub-case 1 -------------------------
        exp = 13384;
        assertEquals(exp, A.infNorm());
    }


    @Test
    void normTestCase() {
        // ------------------------- Sub-case 1 -------------------------
        exp = 13429.354528384523;
        assertEquals(exp, A.norm());
    }
}
