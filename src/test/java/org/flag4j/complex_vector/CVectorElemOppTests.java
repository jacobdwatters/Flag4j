package org.flag4j.complex_vector;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorElemOppTests {

    static Complex128[] aEntries, expEntries;
    static double[] expReEntries;
    static CVector a, exp;
    static Vector expRe;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347), new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345), new Complex128(0, 14.5), new Complex128(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void sqrtTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new Complex128[]{aEntries[0].sqrt(), aEntries[1].sqrt(), aEntries[2].sqrt(),
                aEntries[3].sqrt(), aEntries[4].sqrt()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sqrt());
    }


    @Test
    void absTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        assertEquals(new Vector(
                new double[]{
                        aEntries[0].mag(), aEntries[1].mag(), aEntries[2].mag(),
                        aEntries[3].mag(), aEntries[4].mag()
                }), a.abs());
    }


    @Test
    void recipTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new Complex128[]{aEntries[0].multInv(), aEntries[1].multInv(), aEntries[2].multInv(),
                aEntries[3].multInv(), aEntries[4].multInv()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.recip());
    }
}
