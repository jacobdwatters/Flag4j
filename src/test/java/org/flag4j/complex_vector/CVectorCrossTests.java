package org.flag4j.complex_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorCrossTests {

    CNumber[] aEntries;
    CVectorOld a;

    CNumber[] expEntries;
    CVectorOld exp;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345)};
        a = new CVectorOld(aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        VectorOld b;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new double[]{1.445, -975.25, 0.0024};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[]{
                new CNumber("9016.164088399999-54842.68412499999i"),
                new CNumber("13.355533-95.9620853i"),
                new CNumber("-1405.64562-5974727.13675i")
        };
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[]{1.445, -975.25, 0.0024, 2.45};
        b = new VectorOld(bEntries);
        VectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0)};
        a = new CVectorOld(aEntries);
        bEntries = new double[]{1.445, -975.25, 0.0024};
        b = new VectorOld(bEntries);

        VectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB2));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVectorOld b;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new CNumber[]{
                new CNumber(993.356, 1.6),
                new CNumber(-0.9935, 8.56),
                new CNumber(0, 8.35)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{
                new CNumber("-513.9324125-212.11007574999996i"),
                new CNumber("60428.54887-55858.235232i"),
                new CNumber("-43262.3265585-11026.0765445i")
        };
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new CNumber[]{
                new CNumber(993.356, 1.6),
                new CNumber(-0.9935, 8.56),
                new CNumber(0, 8.35),
                new CNumber(99.24455, 0.0035)};
        b = new CVectorOld(bEntries);
        CVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB));

        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0)};
        a = new CVectorOld(aEntries);
        bEntries = new CNumber[]{
                new CNumber(993.356, 1.6),
                new CNumber(-0.9935, 8.56)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.cross(finalB2));
    }
}
