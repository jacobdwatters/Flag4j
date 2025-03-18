package org.flag4j.numbers.bool_semirings;

import org.flag4j.numbers.BoolSemiring;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConstructorConstantsTests {


    @Test
    void constructorTestCase() {
        BoolSemiring act;
        boolean exp;

        // --------------- sub-case 1 ---------------
        act = new BoolSemiring(true);
        assertTrue(act.getValue());

        // --------------- sub-case 2 ---------------
        act = new BoolSemiring(false);
        assertFalse(act.getValue());

        // --------------- sub-case 3 ---------------
        act = new BoolSemiring(1);
        assertTrue(act.getValue());

        // --------------- sub-case 4 ---------------
        act = new BoolSemiring(0);
        assertFalse(act.getValue());

        // --------------- sub-case 5 ---------------
        assertThrows(IllegalArgumentException.class, () -> new BoolSemiring(2));
        assertThrows(IllegalArgumentException.class, () -> new BoolSemiring(-1));
        assertThrows(IllegalArgumentException.class, () -> new BoolSemiring(1002));
    }


    @Test
    void constantTestCase() {
        // --------------- sub-case 1 ---------------
        assertTrue(BoolSemiring.TRUE.getValue());
        assertTrue(BoolSemiring.ONE.getValue());
        assertTrue(new BoolSemiring(0).getOne().getValue());
        assertTrue(new BoolSemiring(1).isOne());
        assertTrue(new BoolSemiring(0).isZero());
        assertFalse(BoolSemiring.ZERO.getValue());
        assertFalse(BoolSemiring.FALSE.getValue());
        assertFalse(new BoolSemiring(1).getZero().getValue());
    }
}
