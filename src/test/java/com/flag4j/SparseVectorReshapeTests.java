package com.flag4j;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class SparseVectorReshapeTests {

    static SparseVector a;
    SparseVector exp;

    @BeforeAll
    static void setup() {
        double[] values = {1.34, 51.6, -0.00245};
        int[] indices = {0, 5, 103};
        int size = 304;
        a = new SparseVector(size, values, indices);
    }

    @Test
    void reshapeTests() {
        double[] values = {1.34, 51.6, -0.00245};
        int[] indices = {0, 5, 103};
        int size = 304;

        // -------------------- Sub-case 1 --------------------
        exp = new SparseVector(size, values, indices);
        assertEquals(exp, a.reshape(new Shape(size)));

        // -------------------- Sub-case 2 --------------------
        exp = new SparseVector(size, values, indices);
        assertEquals(exp, a.reshape(size));

        // -------------------- Sub-case 3 --------------------
        exp = new SparseVector(size, values, indices);
        assertEquals(exp, a.flatten());

        // -------------------- Sub-case 4 --------------------
        exp = new SparseVector(size, values, indices);
        assertEquals(exp, a.flatten(0));

        // -------------------- Sub-case 5 --------------------
        assertThrows(IllegalArgumentException.class, ()->a.reshape(new Shape(size-3)));

        // -------------------- Sub-case 6 --------------------
        assertThrows(IllegalArgumentException.class, ()->a.reshape(size-3));

        // -------------------- Sub-case 7 --------------------
        assertThrows(AssertionError.class, ()->a.flatten(1));
    }
}
