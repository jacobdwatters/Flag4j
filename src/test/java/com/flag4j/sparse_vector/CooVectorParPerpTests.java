package com.flag4j.sparse_vector;

import com.flag4j.dense.Vector;
import com.flag4j.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CooVectorParPerpTests {

    static double[] aEntries;
    static int[] aIndices;
    static int sparseSize;
    static CooVector a;

    @Test
    void denseParallelTestCase() {
        double[] bEntries;
        Vector b;

        // --------------- Sub-case 1 ---------------
        aEntries = new double[]{1.345, -98.345, 0, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{1.345, 0, -98.345, 0, 0, 24.5, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 2 ---------------
        bEntries = new double[]{1.345*2, 0, -98.345*2, 0, 0, 24.5*2, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 3 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345, 1.345, 0, 24.5, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 4 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 5 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 81;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // --------------- Sub-case 6 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // --------------- Sub-case 7 ---------------
        aEntries = new double[]{-98.345};
        aIndices = new int[]{0};
        sparseSize = 1;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{23.4};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 7 ---------------
        aEntries = new double[]{};
        aIndices = new int[]{};
        sparseSize = 0;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 8 ---------------
        aEntries = new double[]{};
        aIndices = new int[]{};
        sparseSize = 823;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[823];
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 9 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0, 1};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // --------------- Sub-case 10 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/2, 0, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));
    }


    @Test
    void perpTestCase() {
        double[] bEntries;
        Vector b;

        // --------------- Sub-case 1 ---------------
        aEntries = new double[]{1, 1, 1};
        aIndices = new int[]{1, 3, 7};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{1, 0, 1, 0, 1, 1, 1, 0};
        b = new Vector(bEntries);

        assertTrue(a.isPerp(b));

        // --------------- Sub-case 2 ---------------
        aEntries = new double[]{1, 1, 1};
        aIndices = new int[]{1, 3, 7};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{1, 0, 0, 0, 0, 1, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isPerp(b));

        // --------------- Sub-case 3 ---------------
        aEntries = new double[]{1, 1, 1};
        aIndices = new int[]{1, 3, 7};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{1, 1, 0, 0, 0, 1, 0, 0};
        b = new Vector(bEntries);

        assertFalse(a.isPerp(b));

        // --------------- Sub-case 4 ---------------
        aEntries = new double[]{1, 1, 1};
        aIndices = new int[]{1, 3, 7};
        sparseSize = 1845;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{1, 1, 0, 0, 0, 1, 0, 0};
        b = new Vector(bEntries);

        assertFalse(a.isPerp(b));

        // --------------- Sub-case 5 ---------------
        aEntries = new double[]{2.4, -99.24};
        aIndices = new int[]{0, 2};
        sparseSize = 3;
        a = new CooVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{5.3, 349.51145, 106.0/827.0};
        b = new Vector(bEntries);

        assertTrue(a.isPerp(b));
    }
}
