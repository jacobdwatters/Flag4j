package org.flag4j.complex_vector;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorRepeatTests {


    static CVectorOld a;
    static CNumber[] aEntries;
    static CMatrixOld exp;
    static CNumber[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[]{new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)};
        a = new CVectorOld(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.repeat(5, 0));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0, 1)};
        a = new CVectorOld(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(0, 1)},
                {new CNumber(1), new CNumber(0, 1)}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- Sub-case 3 ----------------------
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 0));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[]{new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)};
        a = new CVectorOld(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)},
                {new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                        new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)}
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, a.repeat(5, 1));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0, 1)};
        a = new CVectorOld(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(0, 1)},
                {new CNumber(1), new CNumber(0, 1)}
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, a.repeat(2, 1));
    }
}
