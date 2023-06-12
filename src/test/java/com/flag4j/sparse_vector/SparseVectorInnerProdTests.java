package com.flag4j.sparse_vector;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.common.complex.AggregateComplex;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseVectorInnerProdTests {
    int[] bIndices;
    static int sparseSize;
    static SparseVector a;

    @BeforeAll
    static void setup() {
        double[] aEntries = {1.0, 5.6, -9.355, 215.0};
        int[] aIndices = {1, 2, 8, 13};
        sparseSize = 15;
        a = new SparseVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseInnerProdTest() {
        double[] bEntries;
        SparseVector b;
        double exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{1.34, 55.15, -41.13};
        bIndices = new int[]{0, 2, 8};
        b = new SparseVector(sparseSize, bEntries, bIndices);

        exp = 55.15*5.6 + -9.355*-41.13;

        assertEquals(exp, a.inner(b));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{1.34, 55.15, -41.13};
        bIndices = new int[]{0, 2, 8};
        b = new SparseVector(sparseSize+23, bEntries, bIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test
    void denseInnerProdTest() {
        double[] bEntries;
        Vector b;
        double exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                1.34, 55.15, -41.13, 1, 3.45,
                -99.14, 551.15, 51.5, 0, 0.134,
                0.0245, -0.0, 14.45, 6.133, 4.5};
        b = new Vector(bEntries);

        exp = 55.15 + 5.6*-41.13 + 215.0*6.133;

        assertEquals(exp, a.inner(b));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                1.34, 55.15, -41.13, 1, 3.45,
                -99.14, 551.15, 51.5, 0, 0.134,
                0.0245, -0.0, 14.45};
        b = new Vector(bEntries);
        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test
    void sparseComplexInnerProdTest() {
        CNumber[] bEntries;
        SparseCVector b;
        CNumber exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{new CNumber(1.334, 9.4), new CNumber(-67,14), new CNumber(24,-56.134)};
        bIndices = new int[]{0, 2, 8};
        b = new SparseCVector(sparseSize, bEntries, bIndices);

        exp = AggregateComplex.sum(new CNumber[]{
                new CNumber(-67,14).conj().mult(5.6), new CNumber(24,-56.134).conj().mult(-9.355)
        });

        assertEquals(exp, a.inner(b));


        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{new CNumber(1.334, 9.4), new CNumber(-67,14), new CNumber(24,-56.134)};
        bIndices = new int[]{0, 2, 8};
        b = new SparseCVector(sparseSize-1, bEntries, bIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test
    void denseComplexInnerProdTest() {
        CNumber[] bEntries;
        CVector b;
        CNumber exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(24.1, 54.1), new CNumber(-9.245, 3.4), new CNumber(14.5),
                new CNumber(0, 94.14), new CNumber(), new CNumber(113, 55.62),
                new CNumber(54.13, 5.1), new CNumber(0.0013), new CNumber(-0.924, -994.15),
                new CNumber(24.5516, -0.415), new CNumber(0, 13.46), new CNumber(),
                new CNumber(5.2, 0.924), new CNumber(0.15, .135), new CNumber(25591, 13.5),
                };
        b = new CVector(bEntries);

        exp = AggregateComplex.sum(new CNumber[]{
                new CNumber(-9.245, 3.4).conj().mult(1.0),
                new CNumber(14.5).conj().mult(5.6),
                new CNumber(-0.924, -994.15).conj().mult(-9.355),
                new CNumber(0.15, .135).conj().mult(215.0)
        });

        assertEquals(exp, a.inner(b));


        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(24.1, 54.1), new CNumber(-9.245, 3.4), new CNumber(14.5),
                new CNumber(0, 94.14), new CNumber(), new CNumber(113, 55.62),
                new CNumber(54.13, 5.1), new CNumber(0.0013), new CNumber(-0.924, -994.15),
                new CNumber(24.5516, -0.415), new CNumber(0, 13.46), new CNumber(),
                new CNumber(5.2, 0.924), new CNumber(0.15, .135), new CNumber(25591, 13.5),
                new CNumber(1.15, 4.55), new CNumber(91)
        };
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.inner(finalB));
    }


    @Test
    void normalizeTest() {
        // ----------------------- Sub-case 1 -----------------------
        double mag = Math.sqrt(1.0 + 5.6*5.6 + 9.355*9.355 + 215.0*215.0);
        double[] expEntries = {1.0, 5.6, -9.355, 215.0};
        int[] expIndices = {1, 2, 8, 13};
        SparseVector exp = new SparseVector(sparseSize, expEntries, expIndices).div(mag);

        assertEquals(exp, a.normalize());
    }
}
