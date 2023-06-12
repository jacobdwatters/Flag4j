package com.flag4j.sparse_vector;

import com.flag4j.SparseVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseVectorSetTests {

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
    void setTestCase() {
        double[] values;
        int[] indices;
        int size;

        // -------------------- Sub-case 1 --------------------
        values = new double[]{1.34, 51.6, 22.34, -0.00245};
        indices = new int[]{0, 5, 78, 103};
        size = 304;
        exp = new SparseVector(size, values, indices);

        assertEquals(exp, a.set(22.34, 78));

        // -------------------- Sub-case 2 --------------------
        values = new double[]{44.5, 51.6, -0.00245};
        indices = new int[]{0, 5, 103};
        size = 304;
        exp = new SparseVector(size, values, indices);

        assertEquals(exp, a.set(44.5, 0));
    }
}
