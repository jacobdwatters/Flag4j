package com.flag4j.tensor;


import com.flag4j.Shape;
import com.flag4j.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorSetTests {

    static double[] aEntries, expEntries;
    static Shape shape;
    static Tensor A;

    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234};
        shape = new Shape(1, 3, 2, 1, 2);
        A = new Tensor(shape, aEntries);
    }


    @Test
    void setTest() {
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
}