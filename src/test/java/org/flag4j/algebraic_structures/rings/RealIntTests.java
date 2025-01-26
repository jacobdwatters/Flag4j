package org.flag4j.algebraic_structures.rings;

import org.flag4j.algebraic_structures.RealInt16;
import org.flag4j.algebraic_structures.RealInt32;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RealIntTests {

    @Test
    void addTests() {
        RealInt16 a16;
        RealInt16 b16;
        RealInt16 exp16;

        RealInt32 a32;
        RealInt32 b32;
        RealInt32 exp32;

        // ------------------- sub-case 1 -------------------
        a16 = new RealInt16((short) 1);
        b16 = new RealInt16((short) 152);
        exp16 = new RealInt16((short) 153);

        a32 = new RealInt32(1);
        b32 = new RealInt32(152);
        exp32 = new RealInt32(153);

        assertEquals(exp16, a16.add(b16));
        assertEquals(exp32, a32.add(b32));

        // ------------------- sub-case 2 -------------------
        a16 = new RealInt16((short) 15);
        b16 = new RealInt16((short) 0);
        exp16 = new RealInt16((short) 15);

        a32 = new RealInt32(15);
        b32 = new RealInt32(0);
        exp32 = new RealInt32(15);

        assertEquals(exp16, a16.add(b16));
        assertEquals(exp32, a32.add(b32));

        // ------------------- sub-case 3 -------------------
        a16 = new RealInt16((short) -71);
        b16 = new RealInt16((short) 31);
        exp16 = new RealInt16((short) (-71 + 31));

        a32 = new RealInt32(-71);
        b32 = new RealInt32(31);
        exp32 = new RealInt32(-71 + 31);

        assertEquals(exp16, a16.add(b16));
        assertEquals(exp32, a32.add(b32));
    }


    @Test
    void subTests() {
        RealInt16 a16;
        RealInt16 b16;
        RealInt16 exp16;

        RealInt32 a32;
        RealInt32 b32;
        RealInt32 exp32;

        // ------------------- sub-case 1 -------------------
        a16 = new RealInt16((short) 1);
        b16 = new RealInt16((short) 152);
        exp16 = new RealInt16((short) -151);

        a32 = new RealInt32(1);
        b32 = new RealInt32(152);
        exp32 = new RealInt32(-151);

        assertEquals(exp16, a16.sub(b16));
        assertEquals(exp32, a32.sub(b32));

        // ------------------- sub-case 2 -------------------
        a16 = new RealInt16((short) 15);
        b16 = new RealInt16((short) 0);
        exp16 = new RealInt16((short) 15);

        a32 = new RealInt32(15);
        b32 = new RealInt32(0);
        exp32 = new RealInt32(15);

        assertEquals(exp16, a16.sub(b16));
        assertEquals(exp32, a32.sub(b32));

        // ------------------- sub-case 3 -------------------
        a16 = new RealInt16((short) -71);
        b16 = new RealInt16((short) 31);
        exp16 = new RealInt16((short) (-71 - 31));

        a32 = new RealInt32(-71);
        b32 = new RealInt32(31);
        exp32 = new RealInt32(-71 - 31);

        assertEquals(exp16, a16.sub(b16));
        assertEquals(exp32, a32.sub(b32));
    }


    @Test
    void multTests() {
        RealInt16 a16;
        RealInt16 b16;
        RealInt16 exp16;

        RealInt32 a32;
        RealInt32 b32;
        RealInt32 exp32;

        // ------------------- sub-case 1 -------------------
        a16 = new RealInt16((short) 1);
        b16 = new RealInt16((short) 152);
        exp16 = new RealInt16((short) 152);

        a32 = new RealInt32(1);
        b32 = new RealInt32(152);
        exp32 = new RealInt32(152);

        assertEquals(exp16, a16.mult(b16));
        assertEquals(exp32, a32.mult(b32));

        // ------------------- sub-case 2 -------------------
        a16 = new RealInt16((short) 15);
        b16 = new RealInt16((short) 0);
        exp16 = new RealInt16((short) 0);

        a32 = new RealInt32(15);
        b32 = new RealInt32(0);
        exp32 = new RealInt32(0);

        assertEquals(exp16, a16.mult(b16));
        assertEquals(exp32, a32.mult(b32));

        // ------------------- sub-case 3 -------------------
        a16 = new RealInt16((short) -71);
        b16 = new RealInt16((short) 31);
        exp16 = new RealInt16((short) (-71 * 31));

        a32 = new RealInt32(-71);
        b32 = new RealInt32(31);
        exp32 = new RealInt32(-71 * 31);

        assertEquals(exp16, a16.mult(b16));
        assertEquals(exp32, a32.mult(b32));
    }


    @Test
    void constantTestCase() {
        assertEquals(new RealInt16((short) 1), RealInt16.ONE);
        assertEquals(new RealInt32(1), RealInt32.ONE);
        assertEquals(new RealInt16((short) 0), RealInt16.ZERO);
        assertEquals(new RealInt32(0), RealInt32.ZERO);

        assertTrue(new RealInt16((short) 1).isOne());
        assertFalse(new RealInt16((short) 0).isOne());
        assertFalse(new RealInt16((short) 0).isOne());
        assertTrue(new RealInt16((short) 1).isOne());

        assertTrue(new RealInt32(1).isOne());
        assertFalse(new RealInt32(0).isOne());
        assertFalse(new RealInt32(0).isOne());
        assertTrue(new RealInt32(1).isOne());
    }


    @Test
    void sgnMagTestCase() {
        // Sgn tests
        assertEquals(RealInt16.ONE, RealInt16.sgn(new RealInt16((short) 16)));
        assertEquals(RealInt16.ONE, RealInt16.sgn(new RealInt16((short) 1)));
        assertEquals(RealInt16.ZERO, RealInt16.sgn(new RealInt16((short) 0)));
        assertEquals(RealInt16.NEGATIVE_ONE, RealInt16.sgn(new RealInt16((short) -1)));
        assertEquals(RealInt16.NEGATIVE_ONE, RealInt16.sgn(new RealInt16((short) -251)));

        assertEquals(RealInt32.ONE, RealInt32.sgn(new RealInt32(16)));
        assertEquals(RealInt32.ONE, RealInt32.sgn(new RealInt32(1)));
        assertEquals(RealInt32.ZERO, RealInt32.sgn(new RealInt32(0)));
        assertEquals(RealInt32.NEGATIVE_ONE, RealInt32.sgn(new RealInt32(-1)));
        assertEquals(RealInt32.NEGATIVE_ONE, RealInt32.sgn(new RealInt32(-251)));

        // Magnitude tests
        assertEquals(16.0, new RealInt16((short) 16).mag());
        assertEquals(1.0, new RealInt16((short) 1).mag());
        assertEquals(0.0, new RealInt16((short) 0).mag());
        assertEquals(1.0, new RealInt16((short) -1).mag());
        assertEquals(251.0, new RealInt16((short) -251).mag());

        assertEquals(16.0, new RealInt32(16).mag());
        assertEquals(1.0, new RealInt32(1).mag());
        assertEquals(0.0, new RealInt32(0).mag());
        assertEquals(1.0, new RealInt32(-1).mag());
        assertEquals(251.0, new RealInt32(-251).mag());
    }
}
