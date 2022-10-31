package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberComparisonTest {
    CNumber a;
    CNumber b;
    Object obj;
    Integer integer;
    Double Double;
    double d;

    @Test
    void equalsTest() {
        // ------------ Sub-case 1 ---------------
        a = new CNumber(5.123, 331.4);
        obj = (Object) new CNumber(5.123, 331.4);
        Assertions.assertTrue(a.equals(obj));

        // ------------ Sub-case 2 ---------------
        a = new CNumber(5.123, 331.4);
        b = new CNumber(5.123, 331.4);
        Assertions.assertTrue(a.equals(b));

        // ------------ Sub-case 3 ---------------
        a = new CNumber(5.123, 331.4);
        b = new CNumber(5.123, 331.4000000000001);
        Assertions.assertFalse(a.equals(b));

        // ------------ Sub-case 4 ---------------
        a = new CNumber(5.123, 331.4);
        b = new CNumber(5.123, 331.4000000000001);
        Assertions.assertFalse(a.equals(b));

        // ------------ Sub-case 5 ---------------
        a = new CNumber(-5.123);
        Double = -5.123;
        Assertions.assertTrue(a.equals(Double));

        // ------------ Sub-case 6 ---------------
        a = new CNumber(-5.123, 1);
        Double = -5.123;
        Assertions.assertFalse(a.equals(Double));

        // ------------ Sub-case 7 ---------------
        a = new CNumber(2);
        integer = 2;
        Assertions.assertTrue(a.equals(integer));

        // ------------ Sub-case 8 ---------------
        a = new CNumber(2);
        integer = 0;
        Assertions.assertFalse(a.equals(integer));

        // ------------ Sub-case 9 ---------------
        a = new CNumber(2.09124);
        d = 2.09124;
        Assertions.assertTrue(a.equals(d));

        // ------------ Sub-case 10 ---------------
        a = new CNumber(7);
        d = 7;
        Assertions.assertTrue(a.equals(d));

        // ------------ Sub-case 11 ---------------
        a = new CNumber(3.024);
        d = 2.09124;
        Assertions.assertFalse(a.equals(d));

        // ------------ Sub-case 12 ---------------
        a = new CNumber(9);
        d = 7;
        Assertions.assertFalse(a.equals(d));
    }


    @Test
    void hashCodeTest() {
        a = new CNumber(52344.13, -941.339615);
        int hashPrime = 31;
        int expHash = 7;
        expHash = hashPrime*expHash + Double.hashCode(a.re);
        expHash = hashPrime*expHash + Double.hashCode(a.im);

        Assertions.assertEquals(expHash, a.hashCode());
    }
}
