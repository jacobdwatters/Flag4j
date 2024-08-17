package org.flag4j.complex_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CVectorElemDivTests {
    static CNumber[] aEntries;
    static CVectorOld a;
    CNumber[] expEntries;
    CVectorOld exp;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(4.556, -85.2518), new CNumber(43.1, -99.34551),
                new CNumber(6915.66), new CNumber(0, 9.345)};
        a = new CVectorOld(aEntries);
    }


    /**
     * Checks if all entries of two complex vectors are equal where two NaNs are considered equal.
     * @param exp Expected complex vector.
     * @param act Actual complex vector.
     */
    static void assertEqualsNaN(CVectorOld exp, CVectorOld act) {
        assertEquals(exp.size, act.size);

        for(int i=0; i<exp.size; i++) {
            if(Double.isNaN(exp.get(i).re)) {
                assertTrue(Double.isNaN(act.get(i).re));
            } else {
                assertEquals(exp.get(i).re, act.get(i).re);
            }

            if(Double.isNaN(exp.get(i).im)) {
                assertTrue(Double.isNaN(act.get(i).im));
            } else {
                assertEquals(exp.get(i).im, act.get(i).im);
            }
        }
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        VectorOld b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{2.455, -9.24, 0, 24.50001};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[]{aEntries[0].div(bEntries[0]), aEntries[1].div(bEntries[1]),
                aEntries[2].div(bEntries[2]), aEntries[3].div(bEntries[3])};
        exp = new CVectorOld(expEntries);

        assertEqualsNaN(exp, a.elemDiv(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{2.455, -9.24};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVectorOld b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(-0.00024), new CNumber(0, 85.234),
                new CNumber(0.00234, 15.6), new CNumber(-0.24, 662.115)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{aEntries[0].div(bEntries[0]), aEntries[1].div(bEntries[1]),
                aEntries[2].div(bEntries[2]), aEntries[3].div(bEntries[3])};
        exp = new CVectorOld(expEntries);

        assertEqualsNaN(exp, a.elemDiv(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(0, 85.234),
                new CNumber(0.00234, 15.6), new CNumber(-0.24, 662.115)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }
}
