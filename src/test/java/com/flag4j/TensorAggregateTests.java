package com.flag4j;

import com.flag4j.Shape;
import com.flag4j.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TensorAggregateTests {

    static Shape shape;
    static double[] entries;
    static Tensor A;
    static Double exp;
    static int[] expIndices;

    @BeforeAll
    static void setup() {
        entries = new double[]{1.234, 25.236, 6466.6, -0.0013, -8983.56, -0.01, 1.5, -99.3556, 12.56, 14.5, 11.6, -0.456};
        shape = new Shape(1, 2, 3, 1, 2);
        A = new Tensor(shape, entries);
    }


    @Test
    void minMaxTest() {
        Tensor empty = new Tensor(new Shape(), new double[]{});

        // -------------------------- Minimum Tests --------------------------
        exp = -8983.56;
        assertEquals(exp, A.min());
        exp = 0.0013;
        assertEquals(exp, A.minAbs());
        expIndices = shape.getIndices(4);
        assertArrayEquals(expIndices, A.argMin());

        assertArrayEquals(new int[]{}, empty.argMin());

        // -------------------------- Maximum Tests --------------------------
        exp = 6466.6;
        assertEquals(exp, A.max());
        exp = 8983.56;
        assertEquals(exp, A.maxAbs());
        expIndices = shape.getIndices(2);
        assertArrayEquals(expIndices, A.argMax());

        assertArrayEquals(new int[]{}, empty.argMax());
    }


    @Test
    void sumTest() {
        // -------------------------- Sub-case 1 --------------------------
        exp = 1.234+25.236+6466.6-0.0013-8983.56-0.01+1.5-99.3556+12.56+14.5+11.6-0.456;
        assertEquals(exp, A.sum());
    }
}
