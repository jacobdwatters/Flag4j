package com.flag4j;

import com.flag4j.CustomAssertions;
import com.flag4j.SparseVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseVectorUnaryOpTests {

    static double[] aEntries, expEntries;
    static int[] aIndices, expIndices;
    static int sparseSize;
    static SparseVector a, exp;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.34, -8781.5, 145.4, 6.26, -234.5666, 7.35};
        aIndices = new int[]{0, 1, 6, 44, 78, 80};
        sparseSize = 82;
        a = new SparseVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void absTest() {
        // --------------------- Sub-case 1 ---------------------
        expEntries = new double[]{1.34, 8781.5, 145.4, 6.26, 234.5666, 7.35};
        expIndices = new int[]{0, 1, 6, 44, 78, 80};
        exp = new SparseVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.abs());
    }


    @Test
    void sqrtTest() {
        // --------------------- Sub-case 1 ---------------------
        expEntries = new double[]{Math.sqrt(1.34), Double.NaN, Math.sqrt(145.4),
                Math.sqrt(6.26), Double.NaN, Math.sqrt(7.35)};
        expIndices = new int[]{0, 1, 6, 44, 78, 80};
        exp = new SparseVector(sparseSize, expEntries, expIndices);

        CustomAssertions.assertEqualsNaN(exp, a.sqrt());
    }


    @Test
    void transposeTest() {
        // --------------------- Sub-case 1 ---------------------
        expEntries = new double[]{1.34, -8781.5, 145.4, 6.26, -234.5666, 7.35};
        expIndices = new int[]{0, 1, 6, 44, 78, 80};
        exp = new SparseVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.transpose());
        assertEquals(exp, a.T());
    }


    @Test
    void recipTest() {
        // --------------------- Sub-case 1 ---------------------
        expEntries = new double[]{1.0/1.34, 1.0/-8781.5, 1.0/145.4, 1.0/6.26, 1.0/-234.5666, 1.0/7.35};
        expIndices = new int[]{0, 1, 6, 44, 78, 80};
        exp = new SparseVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.recip());
    }
}
