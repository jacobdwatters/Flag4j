package com.flag4j.complex_vector;

import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CVectorSetTests {
    CNumber[] aEntries, expEntries;
    CVector a, exp;
    int index;

    @Test
    void doubleSetTest() {
        double val;

        // ------------------ Sub-case 1 ------------------
        val = 45.14;
        index = 0;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new CNumber[]{new CNumber(val), new CNumber(9.254), new CNumber(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        val = 45.14;
        index = 1;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(val), new CNumber(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        val = 45.14;
        index = 2;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(val)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 4 ------------------
        val = 45.14;
        index = 3;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);

        double finalVal = val;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->a.set(finalVal, index));

        // ------------------ Sub-case 5 ------------------
        val = 45.14;
        index = -1;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);

        double finalVal2 = val;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->a.set(finalVal2, index));
    }

    @Test
    void CNumberSetTest() {
        CNumber val;

        // ------------------ Sub-case 1 ------------------
        val = new CNumber(2.4567, -9.13357);
        index = 0;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new CNumber[]{val, new CNumber(9.254), new CNumber(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        val = new CNumber(2.4567, -9.13357);
        index = 1;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new CNumber[]{new CNumber(35.632, -8234.6), val, new CNumber(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        val = new CNumber(2.4567, -9.13357);
        index = 2;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), val};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 4 ------------------
        val = new CNumber(2.4567, -9.13357);
        index = 3;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);

        CNumber finalVal = val;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->a.set(finalVal, index));

        // ------------------ Sub-case 5 ------------------
        val = new CNumber(2.4567, -9.13357);
        index = -1;
        aEntries = new CNumber[]{new CNumber(35.632, -8234.6), new CNumber(9.254), new CNumber(0, -824.5)};
        a = new CVector(aEntries);

        CNumber finalVal2 = val;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->a.set(finalVal2, index));
    }
}
