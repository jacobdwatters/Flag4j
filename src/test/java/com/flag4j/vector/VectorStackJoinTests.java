package com.flag4j.vector;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class VectorStackJoinTests {

    double[] aEntries = {1.5, 6.2546, -0.24};
    Vector a = new Vector(aEntries);
    int[] indices;
    int sparseSize;

    @Test
    void realDenseJoinTest() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new Vector(bEntries);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 1.5};
        exp = new Vector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void realSparseJoinTest() {
        double[] bEntries, expEntries;
        SparseVector b;
        Vector exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        sparseSize = 5;
        indices = new int[]{0, 3};
        b = new SparseVector(sparseSize, bEntries, indices);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 0, 0, 1.5, 0};
        exp = new Vector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void complexDenseJoinTest() {
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24),
                new CNumber(1.56, -99345.2), new CNumber("i")};
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void complexSparseJoinTest() {
        CNumber[] bEntries, expEntries;
        SparseCVector b;
        CVector exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        sparseSize = 5;
        indices = new int[]{0, 3};
        b = new SparseCVector(sparseSize, bEntries, indices);
        expEntries = new CNumber[]{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24),
                new CNumber(1.56, -99345.2), new CNumber(), new CNumber(), new CNumber("i"), new CNumber()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }

    // ---------------------------------------------------------------------------

    @Test
    void realDenseStackTest() {
        double[] bEntries;
        Vector b;
        double[][] expEntries;
        Matrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
    }


    @Test
    void realSparseStackTest() {
        double[] bEntries;
        SparseVector b;
        double[][] expEntries;
        Matrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new SparseVector(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0, 0, 0.9345}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new SparseVector(sparseSize, bEntries, indices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
    }


    @Test
    void complexDenseStackTest() {
        CNumber[] bEntries;
        CVector b;
        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {new CNumber(1.56, -99345.2), new CNumber("i"), new CNumber(45, 1.234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
    }


    @Test
    void complexSparseStackTest() {
        CNumber[] bEntries;
        SparseCVector b;
        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new SparseCVector(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {new CNumber(), new CNumber(), new CNumber(-0.234242, 8.1)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new SparseCVector(sparseSize, bEntries, indices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
    }
}
