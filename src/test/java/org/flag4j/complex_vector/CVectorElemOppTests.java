package org.flag4j.complex_vector;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Vector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorElemOppTests {

    static CNumber[] aEntries, expEntries;
    static double[] expReEntries;
    static CVector a, exp;
    static Vector expRe;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void sqrtTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[]{CNumber.sqrt(aEntries[0]), CNumber.sqrt(aEntries[1]), CNumber.sqrt(aEntries[2]),
                CNumber.sqrt(aEntries[3]), CNumber.sqrt(aEntries[4])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sqrt());
    }


    @Test
    void absTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        expReEntries = new double[]{aEntries[0].mag(), aEntries[1].mag(), aEntries[2].mag(),
                aEntries[3].mag(), aEntries[4].mag()};
        expRe = new Vector(expReEntries);
        assertEquals(expRe, a.abs());
    }


    @Test
    void recipTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[]{aEntries[0].multInv(), aEntries[1].multInv(), aEntries[2].multInv(),
                aEntries[3].multInv(), aEntries[4].multInv()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.recip());
    }
}
