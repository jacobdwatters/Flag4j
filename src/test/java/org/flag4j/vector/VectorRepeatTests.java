package org.flag4j.vector;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorRepeatTests {

    static VectorOld a;
    static double[] aEntries;
    static MatrixOld exp;
    static double[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3};
        a = new VectorOld(aEntries);
        expEntries = new double[][]{
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3},
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3},
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3},
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3}
        };
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.repeat(4, 0));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[]{1.123};
        a = new VectorOld(aEntries);
        expEntries = new double[][]{
                {1.123},
                {1.123}
        };
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- Sub-case 3 ----------------------
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 0));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3};
        a = new VectorOld(aEntries);
        expEntries = new double[][]{
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3},
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3},
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3},
                {1.123, 415.125, -0.242424, 150.6, 902.4, 0, 13.3}
        };
        exp = new MatrixOld(expEntries).T();

        assertEquals(exp, a.repeat(4, 1));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[]{1.123};
        a = new VectorOld(aEntries);
        expEntries = new double[][]{
                {1.123},
                {1.123}
        };
        exp = new MatrixOld(expEntries).T();

        assertEquals(exp, a.repeat(2, 1));
    }
}
