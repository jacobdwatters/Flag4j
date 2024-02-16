package com.flag4j.tensor;


import com.flag4j.core.Shape;
import com.flag4j.dense.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class TensorGetSetTests {

    static double[] aEntries, expEntries;
    static Shape shape;
    static Tensor A;

    @BeforeEach
    void setup() {
        aEntries = new double[]{1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234};
        shape = new Shape(1, 3, 2, 1, 2);
        A = new Tensor(shape, aEntries);
    }


    @Test
    void setTestCase() {
        // -------------------- Sub-case 1 --------------------
        A.set(-99.245, 0, 1, 0, 0, 1);
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        expEntries[shape.entriesIndex(0, 1, 0, 0, 1)] = -99.245;

        assertArrayEquals(expEntries, A.entries);

        // -------------------- Sub-case 2 --------------------
        A.set(156.4, 0, 2, 1, 0, 0);
        expEntries = Arrays.copyOf(aEntries, aEntries.length);
        expEntries[shape.entriesIndex(0, 2, 1, 0, 0)] = 156.4;

        assertArrayEquals(expEntries, A.entries);

        // -------------------- Sub-case 3 --------------------
        assertThrows(IllegalArgumentException.class, ()->A.set(156.4, 0, 2, 1));

        // -------------------- Sub-case 4 --------------------
        assertThrows(IllegalArgumentException.class, ()->A.set(156.4, 0, 1, 0, 0, 1, 0));

        // -------------------- Sub-case 5 --------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(-99.245, 0, 0, 0, 0, 2));

        // -------------------- Sub-case 6 --------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(-99.245, 0, 0, -2, 0, 1));
    }


    @Test
    void getTestCase() {
        double exp;

        // ------------------- Sub-case 1 -------------------
        exp = 1.0;
        assertEquals(exp, A.get(0, 2, 1, 0, 0));

        // ------------------- Sub-case 2 -------------------
        exp = 2.556;
        assertEquals(exp, A.get(0, 0, 0, 0, 1));

        // ------------------- Sub-case 3 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(1, 0, 0, 0, 1));

        // ------------------- Sub-case 4 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, -1, 0, 0, 1));

        // ------------------- Sub-case 5 -------------------
        assertThrows(IllegalArgumentException.class, ()->A.get(0, 2, 1, 0));

        // ------------------- Sub-case 6 -------------------
        assertThrows(IllegalArgumentException.class, ()->A.get(0, 2, 1, 0, 1, 0));
    }
}
