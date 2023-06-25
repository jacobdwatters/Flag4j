package com.flag4j.sparse_complex_vector;

import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseCVectorUnaryOpTests {

    static int[] aIndices, expIndices;
    static CNumber[] aEntries, expEntries;
    double[] expEntriesRe;
    static int sparseSize;
    static SparseCVector a, exp;
    SparseVector expRe;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24_023;
        a = new SparseCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void hermTConjTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, 83.6), new CNumber(0, -24.56),
                new CNumber(24.56), new CNumber(-9356.1, -35)
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new SparseCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.H());
        assertEquals(exp, a.hermTranspose());
        assertEquals(exp, a.conj());
    }


    @Test
    void transposeCopyTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new SparseCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.T());
        assertEquals(exp, a.transpose());
        assertEquals(exp, a.copy());
    }


    @Test
    void recipTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6).multInv(), new CNumber(0, 24.56).multInv(),
                new CNumber(24.56).multInv(), new CNumber(-9356.1, 35).multInv()
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new SparseCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.recip());
    }


    @Test
    void absTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        expEntriesRe = new double[]{
                new CNumber(2.455, -83.6).mag(), new CNumber(0, 24.56).mag(),
                new CNumber(24.56).mag(), new CNumber(-9356.1, 35).mag()
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        expRe = new SparseVector(sparseSize, expEntriesRe, expIndices);

        assertEquals(expRe, a.abs());
    }


    @Test
    void sqrtTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        expEntries = new CNumber[]{
                CNumber.sqrt(new CNumber(2.455, -83.6)), CNumber.sqrt(new CNumber(0, 24.56)),
                CNumber.sqrt(new CNumber(24.56)), CNumber.sqrt(new CNumber(-9356.1, 35))
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new SparseCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.sqrt());
    }
}
