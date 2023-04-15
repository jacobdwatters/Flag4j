package com.flag4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorNormTest {

    double exp;
    double[] aEntries;
    Vector a;

    @Test
    void pNormTest() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 12.395642431310288;

        assertEquals(exp, a.norm(2));

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 18.81343;

        assertEquals(exp, a.norm(1));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 9.234000000000005;

        assertEquals(exp, a.norm(234.5));

        // --------------------- Sub-case 6 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 9.234;

        assertEquals(exp, a.norm(Double.POSITIVE_INFINITY));
    }

    @Test
    void infNormTest() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        a = new Vector(aEntries);
        exp = 9.234;

        assertEquals(exp, a.infNorm());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234, 20243234.235, 1119.234, 5.14, -8.234};
        a = new Vector(aEntries);
        exp = 20243234.235;

        assertEquals(exp, a.infNorm());
    }
}
