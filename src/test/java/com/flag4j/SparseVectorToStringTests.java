package com.flag4j;

import com.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseVectorToStringTests {

    String exp;
    static SparseVector a;

    @BeforeAll
    static void setup() {
        double[] nze = {1.34, 525, 63.7, -0.0234};
        int size = 3056;
        int[] indices = {1, 567, 1567, 2506};

        a = new SparseVector(size, nze, indices);
    }


    @AfterAll
    static void cleanup() {
        // Reset all print options.
        PrintOptions.resetAll();
    }


    @Test
    void toStringTest() {
        // --------------------- Sub-case 1 ---------------------
        exp = "Full Shape: 3056\n" +
                "Non-zero entries: [ 1.34  525  63.7  -0.0234 ]\n" +
                "Indices: [1, 567, 1567, 2506]";
        assertEquals(exp, a.toString());

        // --------------------- Sub-case 2 ---------------------
        PrintOptions.setCentering(false);
        PrintOptions.setMaxColumns(2);
        PrintOptions.setPrecision(2);
        exp = "Full Shape: 3056\n" +
                "Non-zero entries: [1.34  ...  -0.02  ]\n" +
                "Indices: [1, 567, 1567, 2506]";
        assertEquals(exp, a.toString());
    }
}
