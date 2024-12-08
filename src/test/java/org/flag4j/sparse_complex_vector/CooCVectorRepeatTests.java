package org.flag4j.sparse_complex_vector;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorRepeatTests {

    static CooCVector a;
    static Complex128[] aEntries;
    static CooCMatrix exp;
    static Complex128[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[]{new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO};
        a = new CVector(aEntries).toCoo();
        expEntries = new Complex128[][]{
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}
        };
        exp = new CMatrix(expEntries).toCoo();

        assertEquals(exp, a.repeat(5, 0));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                Complex128.ZERO, new Complex128(2.45)};
        a = new CVector(aEntries).toCoo();
        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                        Complex128.ZERO, new Complex128(2.45)},
                {Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                        Complex128.ZERO, new Complex128(2.45)}
        };
        exp = new CMatrix(expEntries).toCoo();

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                Complex128.ZERO, new Complex128(2.45)};
        a = new CVector(aEntries).toCoo();

        assertThrows(NegativeArraySizeException.class, ()-> a.repeat(-1, 0));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[]{new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO};
        a = new CVector(aEntries).toCoo();
        expEntries = new Complex128[][]{
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {new Complex128(0.14, 9.2352), Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, new Complex128(0, -1.445), new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}
        };
        exp = new CMatrix(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(5, 1));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                Complex128.ZERO, new Complex128(2.45)};
        a = new CVector(aEntries).toCoo();
        expEntries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                        Complex128.ZERO, new Complex128(2.45)},
                {Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                        Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                        Complex128.ZERO, new Complex128(2.45)}
        };
        exp = new CMatrix(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(2, 1));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new Complex128[]{Complex128.ZERO, Complex128.ZERO, new Complex128(134.4, -51.00024), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, new Complex128(2.45), Complex128.ZERO,
                Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, new Complex128(2.45),
                Complex128.ZERO, new Complex128(2.45)};
        a = new CVector(aEntries).toCoo();

        assertThrows(NegativeArraySizeException.class, ()-> a.repeat(-1, 1));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }
}
