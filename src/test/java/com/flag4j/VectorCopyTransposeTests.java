package com.flag4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorCopyTransposeTests {

    double[] aEntries, expEntries;
    Vector a, exp, act;

    @Test
    void copyTest() {
        aEntries = new double[]{234.45, -0.0234, Double.POSITIVE_INFINITY, Double.NaN, -0.0, 1};
        a = new Vector(aEntries);
        expEntries = new double[]{234.45, -0.0234, Double.POSITIVE_INFINITY, Double.NaN, -0.0, 1};
        exp = new Vector(expEntries);

        // ------------------ Sub-case 1 ------------------
        act = a.copy();

        for(int i=0; i<exp.size; i++) {
            if(Double.isNaN(exp.get(i))) {
                assertTrue(Double.isNaN(exp.get(i)));
            } else {
                assertEquals(exp.get(i), act.get(i));
            }
        }


        // ------------------ Sub-case 2 ------------------
        act = a.T();

        for(int i=0; i<exp.size; i++) {
            if(Double.isNaN(exp.get(i))) {
                assertTrue(Double.isNaN(exp.get(i)));
            } else {
                assertEquals(exp.get(i), act.get(i));
            }
        }


        // ------------------ Sub-case 3 ------------------
        act = a.transpose();

        for(int i=0; i<exp.size; i++) {
            if(Double.isNaN(exp.get(i))) {
                assertTrue(Double.isNaN(exp.get(i)));
            } else {
                assertEquals(exp.get(i), act.get(i));
            }
        }
    }
}
