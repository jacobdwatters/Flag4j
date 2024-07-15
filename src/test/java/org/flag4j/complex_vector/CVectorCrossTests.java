package org.flag4j.complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorCrossTests {

    CNumber[] aEntries;
    CVector a;

    CNumber[] expEntries;
    CVector exp;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345)};
        a = new CVector(aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new double[]{1.445, -975.25, 0.0024};
        b = new Vector(bEntries);
        expEntries = new CNumber[]{
                new CNumber("9016.164088399999-54842.68412499999i"),
                new CNumber("13.355533-95.9620853i"),
                new CNumber("-1405.64562-5974727.13675i")
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[]{1.445, -975.25, 0.0024, 2.45};
        b = new Vector(bEntries);
        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0)};
        a = new CVector(aEntries);
        bEntries = new double[]{1.445, -975.25, 0.0024};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB2));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVector b;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new CNumber[]{
                new CNumber(993.356, 1.6),
                new CNumber(-0.9935, 8.56),
                new CNumber(0, 8.35)};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{
                new CNumber("-513.9324125-212.11007574999996i"),
                new CNumber("60428.54887-55858.235232i"),
                new CNumber("-43262.3265585-11026.0765445i")
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new CNumber[]{
                new CNumber(993.356, 1.6),
                new CNumber(-0.9935, 8.56),
                new CNumber(0, 8.35),
                new CNumber(99.24455, 0.0035)};
        b = new CVector(bEntries);
        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0)};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{
                new CNumber(993.356, 1.6),
                new CNumber(-0.9935, 8.56)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB2));
    }
}
