package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorElemMultTests {

    static CNumber[] aEntries;
    static CVector a;
    CNumber[] expEntries;
    CVector exp;

    int[] sparseIndices;
    int sparseSize;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(4.556, -85.2518), new CNumber(43.1, -99.34551),
                new CNumber(6915.66), new CNumber(0, 9.345)};
        a = new CVector(aEntries);
    }


    @Test
    void realDenseTest() {
        double[] bEntries;
        Vector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{2.455, -9.24, 0, 24.50001};
        b = new Vector(bEntries);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                aEntries[2].mult(bEntries[2]), aEntries[3].mult(bEntries[3])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{2.455, -9.24};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void realSparseTest() {
        double[] bEntries;
        SparseVector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{0, 1};
        sparseSize = 4;
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                new CNumber(), new CNumber()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{1, 3};
        sparseSize = 4;
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{new CNumber(), aEntries[1].mult(bEntries[0]),
                new CNumber(), aEntries[3].mult(bEntries[1])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 3 -------------------
        bEntries = new double[]{2.455, 24.50001};
        sparseIndices = new int[]{0, 1};
        sparseSize = 185234;
        b = new SparseVector(sparseSize, bEntries, sparseIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexDenseTest() {
        CNumber[] bEntries;
        CVector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(-0.00024), new CNumber(0, 85.234),
            new CNumber(0.00234, 15.6), new CNumber(-0.24, 662.115)};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                aEntries[2].mult(bEntries[2]), aEntries[3].mult(bEntries[3])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(0, 85.234),
                new CNumber(0.00234, 15.6), new CNumber(-0.24, 662.115)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }

    @Test
    void complexSparseTest() {
        CNumber[] bEntries;
        SparseCVector b;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(234.566, -9.225), new CNumber(0.00024, 15.5)};
        sparseIndices = new int[]{0, 1};
        sparseSize = 4;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]),
                new CNumber(), new CNumber()};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(-23.566, 0), new CNumber(0, 15.5)};
        sparseIndices = new int[]{1, 3};
        sparseSize = 4;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{new CNumber(), aEntries[1].mult(bEntries[0]),
                new CNumber(), aEntries[3].mult(bEntries[1])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------- Sub-case 3 -------------------
        bEntries = new CNumber[]{new CNumber(-23.566, 0), new CNumber(0, 15.5)};
        sparseIndices = new int[]{0, 1};
        sparseSize = 185234;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }
}
