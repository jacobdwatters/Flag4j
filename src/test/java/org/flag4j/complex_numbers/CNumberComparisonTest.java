package org.flag4j.complex_numbers;

import org.flag4j.algebraic_structures.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128ComparisonTest {
    Complex128 a;
    Complex128 b;
    Object obj;
    Integer integer;
    Double Double;
    double d;
    int compare;
    int expCompare;

    @Test
    void equalsTestCase() {
        // ------------ Sub-case 1 ---------------
        a = new Complex128(5.123, 331.4);
        obj = new Complex128(5.123, 331.4);
        Assertions.assertEquals(a, obj);

        // ------------ Sub-case 2 ---------------
        a = new Complex128(5.123, 331.4);
        b = new Complex128(5.123, 331.4);
        Assertions.assertEquals(a, b);

        // ------------ Sub-case 3 ---------------
        a = new Complex128(5.123, 331.4);
        b = new Complex128(5.123, 331.4000000000001);
        Assertions.assertNotEquals(a, b);

        // ------------ Sub-case 4 ---------------
        a = new Complex128(5.123, 331.4);
        b = new Complex128(5.123, 331.4000000000001);
        Assertions.assertNotEquals(a, b);

        // ------------ Sub-case 5 ---------------
        a = new Complex128(-5.123);
        Double = -5.123;
        Assertions.assertEquals(a, new Complex128(Double));

        // ------------ Sub-case 6 ---------------
        a = new Complex128(-5.123, 1);
        Double = -5.123;
        Assertions.assertNotEquals(a, new Complex128(Double));

        // ------------ Sub-case 7 ---------------
        a = new Complex128(2);
        integer = 2;
        Assertions.assertEquals(a, new Complex128(integer));

        // ------------ Sub-case 8 ---------------
        a = new Complex128(2);
        integer = 0;
        Assertions.assertNotEquals(a, new Complex128(integer));

        // ------------ Sub-case 9 ---------------
        a = new Complex128(2.09124);
        d = 2.09124;
        Assertions.assertTrue(a.equals(d));

        // ------------ Sub-case 10 ---------------
        a = new Complex128(7);
        d = 7;
        Assertions.assertTrue(a.equals(d));

        // ------------ Sub-case 11 ---------------
        a = new Complex128(3.024);
        d = 2.09124;
        Assertions.assertFalse(a.equals(d));

        // ------------ Sub-case 12 ---------------
        a = new Complex128(9);
        d = 7;
        Assertions.assertFalse(a.equals(d));
    }


    @Test
    void hashCodeTestCase() {
        a = new Complex128(52344.13, -941.339615);
        int hashPrime = 31;
        int expHash = 17;
        expHash = hashPrime*expHash + java.lang.Double.hashCode(a.re);
        expHash = hashPrime*expHash + java.lang.Double.hashCode(a.im);

        Assertions.assertEquals(expHash, a.hashCode());
    }


    @Test
    void compareTestCase() {
        // -------- Sub-case 1 ----------
        a = new Complex128(400);
        b = new Complex128(3, 1.21);
        expCompare = 1;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);

        // -------- Sub-case 2 ----------
        a = new Complex128(4);
        b = new Complex128(3, 3.4214);
        expCompare = -1;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);

        // -------- Sub-case 3 ----------
        a = new Complex128(34.13, 11.44);
        b = new Complex128(11.44, 34.13);
        expCompare = 0;

        compare = a.compareTo(b);
        Assertions.assertEquals(expCompare, compare);
    }
}
