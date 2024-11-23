package org.flag4j.complex_vector;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorCrossTests {

    Complex128[] aEntries;
    CVector a;

    Complex128[] expEntries;
    CVector exp;

    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347),
                new Complex128(-9.234, 5.0),
                new Complex128(9.245, -56.2345)};
        a = new CVector(aEntries);
    }


    @Test
    void complexDenseTestCase() {
        Complex128[] bEntries;
        CVector b;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new Complex128[]{
                new Complex128(993.356, 1.6),
                new Complex128(-0.9935, 8.56),
                new Complex128(0, 8.35)};
        b = new CVector(bEntries);
        expEntries = new Complex128[]{
                new Complex128("-513.9324125-212.11007574999996i"),
                new Complex128("60428.54887-55858.235232i"),
                new Complex128("-43262.3265585-11026.0765445i")
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new Complex128[]{
                new Complex128(993.356, 1.6),
                new Complex128(-0.9935, 8.56),
                new Complex128(0, 8.35),
                new Complex128(99.24455, 0.0035)};
        b = new CVector(bEntries);
        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new Complex128[]{new Complex128(1.455, 6126.347),
                new Complex128(-9.234, 5.0)};
        a = new CVector(aEntries);
        bEntries = new Complex128[]{
                new Complex128(993.356, 1.6),
                new Complex128(-0.9935, 8.56)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB2));
    }
}
