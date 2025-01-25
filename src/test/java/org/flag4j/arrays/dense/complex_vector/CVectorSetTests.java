package org.flag4j.arrays.dense.complex_vector;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorSetTests {
    Complex128[] aEntries, expEntries;
    CVector a, exp;
    int index;

    @Test
    void doubleSetTestCase() {
        double val;

        // ------------------ Sub-case 1 ------------------
        val = 45.14;
        index = 0;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(val), new Complex128(9.254), new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        val = 45.14;
        index = 1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(val), new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        val = 45.14;
        index = 2;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(val)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 4 ------------------
        val = 45.14;
        index = 3;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        double finalVal = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal, index));

        // ------------------ Sub-case 5 ------------------
        val = 45.14;
        index = -1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        double finalVal2 = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal2, index));
    }

    @Test
    void Complex128SetTestCase() {
        Complex128 val;

        // ------------------ Sub-case 1 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 0;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{val, new Complex128(9.254), new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), val, new Complex128(0, -824.5)};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 2;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);
        expEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), val};
        exp = new CVector(expEntries);

        a.set(val, index);

        assertEquals(exp, a);

        // ------------------ Sub-case 4 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = 3;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        Complex128 finalVal = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal, index));

        // ------------------ Sub-case 5 ------------------
        val = new Complex128(2.4567, -9.13357);
        index = -1;
        aEntries = new Complex128[]{new Complex128(35.632, -8234.6), new Complex128(9.254), new Complex128(0, -824.5)};
        a = new CVector(aEntries);

        Complex128 finalVal2 = val;
        assertThrows(IndexOutOfBoundsException.class, ()->a.set(finalVal2, index));
    }
}
