package org.flag4j.vector;

import org.flag4j.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorAggregateTests {

    Vector a;
    double[] aEntries;
    Double expAggregate;
    int[] expArg;

    @Test
    void sumTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -194.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expAggregate = 1.45 - 194.5666 + 430.5 + 563.3;

        assertEquals(expAggregate, a.sum());
    }


    @Test
    void minTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -194.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expAggregate = -194.5666;

        assertEquals(expAggregate, a.min());
    }


    @Test
    void minAbsTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -194.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expAggregate = 1.45;

        assertEquals(expAggregate, a.minAbs());
    }


    @Test
    void argMinTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -194.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expArg = new int[]{1};

        assertArrayEquals(expArg, a.argMin());
    }


    @Test
    void maxTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -194.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expAggregate = 563.3;

        assertEquals(expAggregate, a.max());
    }


    @Test
    void maxAbsTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -1934.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expAggregate = 1934.5666;

        assertEquals(expAggregate, a.maxAbs());
    }


    @Test
    void argMaxTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.45, -194.5666, 430.5, 563.3};
        a = new Vector(aEntries);
        expArg = new int[]{3};

        assertArrayEquals(expArg, a.argMax());

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[]{1.45, -194.5666, 4301.5, 563.3};
        a = new Vector(aEntries);
        expArg = new int[]{2};

        assertArrayEquals(expArg, a.argMax());
    }
}
