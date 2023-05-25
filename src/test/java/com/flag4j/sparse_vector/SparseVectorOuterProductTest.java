package com.flag4j.sparse_vector;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseVectorOuterProductTest {

    int[] bIndices;
    static int sparseSize;
    static SparseVector a;

    @BeforeAll
    static void setup() {
        double[] aEntries = {1.0, 5.6, -9.355};
        int[] aIndices = {1, 2, 4};
        sparseSize = 5;
        a = new SparseVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseOuterProdTest() {
        double[] bEntries;
        double[][] expEntries;
        SparseVector b;
        Matrix exp;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.34, -99.4};
        bIndices = new int[]{0, 2};
        b = new SparseVector(sparseSize, bEntries, bIndices);
        expEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {1.34, 0.0, -99.4, 0.0, 0.0},
                {7.504, 0.0, -556.64, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {-12.535700000000002, 0.0, 929.8870000000001, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.34, -99.4};
        bIndices = new int[]{0, 2};
        b = new SparseVector(sparseSize+1445, bEntries, bIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.outer(finalB));
    }


    @Test
    void sparseComplexOuterProdTest() {
        CNumber[] bEntries;
        CNumber[][] expEntries;
        SparseCVector b;
        CMatrix exp;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(1.34, 0.0244), new CNumber(-99, 815.66)};
        bIndices = new int[]{0, 2};
        b = new SparseCVector(sparseSize, bEntries, bIndices);
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.34+0.0244i"), new CNumber("0.0"), new CNumber("-99.0+815.66i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("7.504+0.13664i"), new CNumber("0.0"), new CNumber("-554.4+4567.696i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-12.535700000000002-0.22826200000000002i"), new CNumber("0.0"), new CNumber("926.1450000000001-7630.4993i"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(1.34, 0.0244), new CNumber(-99, 815.66)};
        bIndices = new int[]{0, 2};
        b = new SparseCVector(sparseSize+103, bEntries, bIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.outer(finalB));
    }
}
