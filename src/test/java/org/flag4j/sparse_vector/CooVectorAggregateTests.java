package org.flag4j.sparse_vector;

import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.linalg.VectorNorms;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CooVectorAggregateTests {

    static double[] aValues;
    static int[] aIndices;
    static int size;
    static CooVectorOld a;


    @BeforeAll
    static void setup() {
        aValues = new double[]{1.34, 51.6, -0.00245};
        aIndices = new int[]{0, 5, 103};
        size = 304;
        a = new CooVectorOld(size, aValues, aIndices);
    }


    @Test
    void sumTestCase() {
        double exp;

        // --------------------- Sub-case 1 ---------------------
        exp = 1.34+51.6-0.00245;
        assertEquals(exp, a.sum());
    }


    @Test
    void argminMaxTestCase() {
        int[] exp;

        // --------------------- Sub-case 1 ---------------------
        exp = new int[]{5};
        assertArrayEquals(exp, a.argmax());

        exp = new int[]{103};
        assertArrayEquals(exp, a.argmin());
    }


    @Test
    void normTestCase() {
        double exp;

        // --------------------- Sub-case 1 ---------------------
        exp = 51.61739635047955;
        assertEquals(exp, VectorNorms.norm(a));

        // --------------------- Sub-case 2 ---------------------
        exp = 51.82204923335818;
        assertEquals(exp, VectorNorms.norm(a, 1.4));

        // --------------------- Sub-case 3 ---------------------
        exp = 51.599999999999994;
        assertEquals(exp, VectorNorms.norm(a, 23));

        // --------------------- Sub-case 4 ---------------------
        exp = 152.7777441673176;
        assertEquals(exp, VectorNorms.norm(a, 0.3));

        // --------------------- Sub-case 5 ---------------------
        exp = 51.6;
        assertEquals(exp, VectorNorms.norm(a, Double.POSITIVE_INFINITY));

        // --------------------- Sub-case 6 ---------------------
        exp = 51.6;
        assertEquals(exp, VectorNorms.infNorm(a));
    }
}
