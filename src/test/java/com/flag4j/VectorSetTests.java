package com.flag4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorSetTests {

    double[] entries, expEntries;
    Vector a, exp;

    @Test
    void setTest() {
        // --------------------- Sub-case 1 ---------------------
        entries = new double[]{1.34, -99.345, 1345.255, 1.5};
        a = new Vector(entries);
        expEntries = new double[]{-0.0009843, -99.345, -14.5, 1.5};
        exp = new Vector(expEntries);

        a.set(-0.0009843, 0);
        a.set(-14.5, 2);

        assertEquals(exp, a);
    }
}
