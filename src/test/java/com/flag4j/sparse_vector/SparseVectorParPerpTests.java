package com.flag4j.sparse_vector;

import com.flag4j.SparseVector;
import com.flag4j.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SparseVectorParPerpTests {

    static double[] aEntries;
    static int[] aIndices;
    static int sparseSize;
    static SparseVector a;

    @Test
    void denseParallelTest() {
        double[] bEntries;
        Vector b;

        // --------------- Sub-case 1 ---------------
        aEntries = new double[]{1.345, -98.345, 0, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new SparseVector(sparseSize, aEntries, aIndices);
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
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345, 1.345, 0, 24.5, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 4 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0, 0};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 5 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 81;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // --------------- Sub-case 6 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // --------------- Sub-case 7 ---------------
        aEntries = new double[]{-98.345};
        aIndices = new int[]{0};
        sparseSize = 1;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{23.4};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 7 ---------------
        aEntries = new double[]{};
        aIndices = new int[]{};
        sparseSize = 0;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{};
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 8 ---------------
        aEntries = new double[]{};
        aIndices = new int[]{};
        sparseSize = 823;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[823];
        b = new Vector(bEntries);

        assertTrue(a.isParallel(b));

        // --------------- Sub-case 9 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/345.965178, 0, 1};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));

        // --------------- Sub-case 10 ---------------
        aEntries = new double[]{0, -98.345, 1.345, 24.5};
        aIndices = new int[]{0, 2, 3, 5};
        sparseSize = 8;
        a = new SparseVector(sparseSize, aEntries, aIndices);
        bEntries = new double[]{0, 0, -98.345/345.965178, 1.345/345.965178, 0, 24.5/2, 0, 0};
        b = new Vector(bEntries);

        assertFalse(a.isParallel(b));
    }
}
