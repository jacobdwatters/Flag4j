package com.flag4j.util;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RandomCNumberTests {
    RandomCNumber rng;
    CNumber a;

    @Test
    void constructorTests() {
        rng = new RandomCNumber();
        assertFalse(rng==null);

        rng = new RandomCNumber(42l);
        assertFalse(rng==null);
    }


    @Test
    void randomTest() {
        rng = new RandomCNumber();

        a = rng.randn();
        assertNotNull(a);
        assertEquals(0, a.im);

        a = rng.randn(true);
        assertNotNull(a);

        a = rng.random();
        assertNotNull(a);
        assertTrue(a.re<1);
        assertTrue(a.re>=0);
        assertEquals(0, a.im);

        a = rng.random(3);
        assertNotNull(a);
        assertTrue(Math.sqrt(a.re*a.re + a.im*a.im)<3.1);
        assertTrue(Math.sqrt(a.re*a.re + a.im*a.im)>2.9);

        assertThrows(IllegalArgumentException.class, () -> rng.random(-1));

        a = rng.random(-1, 2);
        assertTrue(a.re<2.1);
        assertTrue(a.re>=-1.1);
        assertEquals(0, a.im);

        a = rng.random(14, 15);
        assertTrue(a.re<15.1);
        assertTrue(a.re>=13.9);
        assertEquals(0, a.im);

        assertThrows(IllegalArgumentException.class, () -> rng.random(15, 14));

        a = rng.random(1, 2, true);
        assertTrue(Math.sqrt(a.re*a.re + a.im*a.im)<2);
        assertTrue(Math.sqrt(a.re*a.re + a.im*a.im)>=1);

        a = rng.random(14, 15, true);
        assertTrue(Math.sqrt(a.re*a.re + a.im*a.im)<15);
        assertTrue(Math.sqrt(a.re*a.re + a.im*a.im)>=14);

        assertThrows(IllegalArgumentException.class, () -> rng.random(-1, 2, true));
        assertThrows(IllegalArgumentException.class, () -> rng.random(15, 14, true));
        assertThrows(IllegalArgumentException.class, () -> rng.random(15, 14, true, false));
    }
}
