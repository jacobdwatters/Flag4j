package com.flag4j.complex_vector;

import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorScalMultDivTests {

    static CNumber[] aEntries;
    static CVector a;
    CNumber[] expEntries;
    CVector exp;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356)};
        a = new CVector(aEntries);
    }


    @Test
    void realScalMultTestCase() {
        double b;

        // ------------------- Sub-case 1 -------------------
        b = 129.12354;
        expEntries = new CNumber[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));

        // ------------------- Sub-case 2 -------------------
        b = -9.12354;
        expEntries = new CNumber[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));
    }


    @Test
    void complexScalMultTestCase() {
        CNumber b;

        // ------------------- Sub-case 1 -------------------
        b = new CNumber(-99.234, 56.1);
        expEntries = new CNumber[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));

        // ------------------- Sub-case 2 -------------------
        b = new CNumber(9.234000014, -56.1);
        expEntries = new CNumber[]{aEntries[0].mult(b), aEntries[1].mult(b), aEntries[2].mult(b), aEntries[3].mult(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.mult(b));
    }


    @Test
    void realScalDivTestCase() {
        double b;

        // ------------------- Sub-case 1 -------------------
        b = 129.12354;
        expEntries = new CNumber[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));

        // ------------------- Sub-case 2 -------------------
        b = -9.12354;
        expEntries = new CNumber[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));
    }


    @Test
    void complexScalDivTestCase() {
        CNumber b;

        // ------------------- Sub-case 1 -------------------
        b = new CNumber(-99.234, 56.1);
        expEntries = new CNumber[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));

        // ------------------- Sub-case 2 -------------------
        b = new CNumber(9.234000014, -56.1);
        expEntries = new CNumber[]{aEntries[0].div(b), aEntries[1].div(b), aEntries[2].div(b), aEntries[3].div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.div(b));
    }
}
