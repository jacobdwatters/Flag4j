package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorElemOppTests {

    static CNumber[] aEntries, expEntries;
    static CVector a, exp;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void sqrtTest() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[]{CNumber.sqrt(aEntries[0]), CNumber.sqrt(aEntries[1]), CNumber.sqrt(aEntries[2]),
                CNumber.sqrt(aEntries[3]), CNumber.sqrt(aEntries[4])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.sqrt());
    }


    @Test
    void absTest() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[]{aEntries[0].mag(), aEntries[1].mag(), aEntries[2].mag(),
                aEntries[3].mag(), aEntries[4].mag()};
        exp = new CVector(expEntries);
        assertEquals(exp, a.abs());
    }


    @Test
    void recipTest() {
        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[]{aEntries[0].addInv(), aEntries[1].addInv(), aEntries[2].addInv(),
                aEntries[3].addInv(), aEntries[4].addInv()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.recip());
    }
}
