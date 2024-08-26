package org.flag4j.sparse_vector;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorRepeatTests {

    static CooVectorOld a;
    static double[] aEntries;
    static CooMatrixOld exp;
    static double[][] expEntries;

    @Test
    void repeatRowTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45};
        a = new VectorOld(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45}
        };
        exp = new MatrixOld(expEntries).toCoo();

        assertEquals(exp, a.repeat(4, 0));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new VectorOld(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45},
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45}
        };
        exp = new MatrixOld(expEntries).toCoo();

        assertEquals(exp, a.repeat(2, 0));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new VectorOld(aEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 0));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }


    @Test
    void repeatColTest() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[]{0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45};
        a = new VectorOld(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45},
                {0, 0, 0, -15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.45}
        };
        exp = new MatrixOld(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(4, 1));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new VectorOld(aEntries).toCoo();
        expEntries = new double[][]{
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45},
                {0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45}
        };
        exp = new MatrixOld(expEntries).T().toCoo();

        assertEquals(exp, a.repeat(2, 1));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[]{0, 0, 1.324, -15.2, 0, 0, 0, 0, 0, 1400.245, -.241, 0.0024, 0, 0, 0, 2.45};
        a = new VectorOld(aEntries).T().toCoo();

        assertThrows(IllegalArgumentException.class, ()-> a.repeat(-1, 1));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, -2));
        assertThrows(IllegalArgumentException.class, ()-> a.repeat(13, 2));
    }
}
