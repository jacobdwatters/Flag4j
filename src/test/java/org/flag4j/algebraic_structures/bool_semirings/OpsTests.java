package org.flag4j.algebraic_structures.bool_semirings;

import org.flag4j.algebraic_structures.BoolSemiring;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class OpsTests {

    @Test
    void orTestCase() {
        BoolSemiring a;
        BoolSemiring b;
        BoolSemiring exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(true);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(false);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(true);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));

        // --------------- sub-case 4 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(false);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.or(b));
        assertEquals(exp, a.add(b));
    }


    @Test
    void xorTestCase() {
        BoolSemiring a;
        BoolSemiring b;
        BoolSemiring exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(true);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.xor(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(false);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.xor(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(true);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.xor(b));

        // --------------- sub-case 4 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(false);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.xor(b));
    }


    @Test
    void andTestCase() {
        BoolSemiring a;
        BoolSemiring b;
        BoolSemiring exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(true);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(false);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(true);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));

        // --------------- sub-case 4 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(false);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.and(b));
        assertEquals(exp, a.mult(b));
    }


    @Test
    void notTestCase() {
        BoolSemiring a;
        BoolSemiring exp;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiring(true);
        exp = new BoolSemiring(false);

        assertEquals(exp, a.not());

        // --------------- sub-case 2 ---------------
        a = new BoolSemiring(false);
        exp = new BoolSemiring(true);

        assertEquals(exp, a.not());
    }


    @Test
    void equalsCompareTestCase() {
        BoolSemiring a;
        BoolSemiring b;

        // --------------- sub-case 1 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(true);

        assertTrue(a.equals(b));
        assertTrue(a.equals(a));
        assertFalse(a.equals(null));
        assertEquals(0, a.compareTo(b));

        // --------------- sub-case 2 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(false);

        assertTrue(a.equals(b));
        assertTrue(a.equals(a));
        assertFalse(a.equals(null));
        assertEquals(0, a.compareTo(b));

        // --------------- sub-case 3 ---------------
        a = new BoolSemiring(true);
        b = new BoolSemiring(false);

        assertFalse(a.equals(b));
        assertTrue(a.compareTo(b) > 0);

        // --------------- sub-case 4 ---------------
        a = new BoolSemiring(false);
        b = new BoolSemiring(true);

        assertFalse(a.equals(b));
        assertTrue(a.compareTo(b) < 0);
    }
}
