package org.flag4j.sparse_complex_vector;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorRepeatTests {

    static CooCVectorOld a;
    static CNumber[] aEntries;
    static CooCMatrixOld exp;
    static CNumber[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[]{new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO};
        a = new CVectorOld(aEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}
        };
        exp = new CMatrixOld(expEntries).toCoo();

        assertEquals(exp, a.repeat(5, 0));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[]{CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                CNumber.ZERO, new CNumber(2.45)};
        a = new CVectorOld(aEntries).toCoo();
        expEntries = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                        CNumber.ZERO, new CNumber(2.45)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                        CNumber.ZERO, new CNumber(2.45)}
        };
        exp = new CMatrixOld(expEntries).toCoo();

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[]{CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                CNumber.ZERO, new CNumber(2.45)};
        a = new CVectorOld(aEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 0));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[]{new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO};
        a = new CVectorOld(aEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0.14, 9.2352), CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, new CNumber(0, -1.445), new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}
        };
        exp = new CMatrixOld(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(5, 1));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[]{CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                CNumber.ZERO, new CNumber(2.45)};
        a = new CVectorOld(aEntries).toCoo();
        expEntries = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                        CNumber.ZERO, new CNumber(2.45)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                        CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                        CNumber.ZERO, new CNumber(2.45)}
        };
        exp = new CMatrixOld(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(2, 1));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[]{CNumber.ZERO, CNumber.ZERO, new CNumber(134.4, -51.00024), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(2.45), CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(2.45),
                CNumber.ZERO, new CNumber(2.45)};
        a = new CVectorOld(aEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 1));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }
}
