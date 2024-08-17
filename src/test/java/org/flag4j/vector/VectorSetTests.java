package org.flag4j.vector;

import org.flag4j.arrays_old.dense.VectorOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorSetTests {

    double[] entries, expEntries;
    VectorOld a, exp;

    @Test
    void setTestCase() {
        // --------------------- Sub-case 1 ---------------------
        entries = new double[]{1.34, -99.345, 1345.255, 1.5};
        a = new VectorOld(entries);
        expEntries = new double[]{-0.0009843, -99.345, -14.5, 1.5};
        exp = new VectorOld(expEntries);

        a.set(-0.0009843, 0);
        a.set(-14.5, 2);

        assertEquals(exp, a);
    }
}
