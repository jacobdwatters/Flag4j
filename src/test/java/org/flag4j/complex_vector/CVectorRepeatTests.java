package org.flag4j.complex_vector;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorRepeatTests {


    static CVector a;
    static CNumber[] aEntries;
    static CMatrix exp;
    static CNumber[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[]{new CNumber(20.24, 1), new CNumber(-0.1451), new CNumber(93.1, 515.3),
                new CNumber(0, -9.245), new CNumber(234.1), new CNumber(1800.24, -9923001.4)};
        a = new CVector(aEntries);
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
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.repeat(5, 0));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0, 1)};
        a = new CVector(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(0, 1)},
                {new CNumber(1), new CNumber(0, 1)}
        };
        exp = new CMatrix(expEntries);

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
        a = new CVector(aEntries);
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
        exp = new CMatrix(expEntries).T();

        assertEquals(exp, a.repeat(5, 1));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0, 1)};
        a = new CVector(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(0, 1)},
                {new CNumber(1), new CNumber(0, 1)}
        };
        exp = new CMatrix(expEntries).T();

        assertEquals(exp, a.repeat(2, 1));
    }
}
