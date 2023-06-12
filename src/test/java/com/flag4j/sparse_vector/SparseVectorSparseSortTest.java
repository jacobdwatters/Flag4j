package com.flag4j.sparse_vector;

import com.flag4j.SparseVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseVectorSparseSortTest {
    static int[] aIndices, expIndices;
    static double[] aEntries, expEntries;
    static int sparseSize;
    static SparseVector a, exp;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.345, -989.234, 5.15, 617.4, 1.34, 5126.234, 456.3};
        aIndices = new int[]{36, 13, 4, 11345, 3, 645, 3324};
        sparseSize = 24_023;
        a = new SparseVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseSortTest() {
        // --------------------- Sub-case 1 ---------------------
        expIndices = new int[]{3, 4, 13, 36, 645, 3324, 11345};
        expEntries = new double[]{1.34, 5.15, -989.234, 1.345, 5126.234, 456.3, 617.4};
        exp = new SparseVector(sparseSize, expEntries, expIndices);

        a.sparseSort();
        assertEquals(exp, a);
    }
}
