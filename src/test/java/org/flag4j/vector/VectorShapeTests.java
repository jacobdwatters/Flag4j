package org.flag4j.vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorShapeTests {

    int[] indices;
    int size;

    double[] aEntries;
    VectorOld A;

    @Test
    void sizeTestCase() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234};
        A = new VectorOld(aEntries);

        assertEquals(aEntries.length, A.length());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[]{1.43543, 8.144, -9.234, 20243234.235, 1119.234, 5.14, -8.234};
        A = new VectorOld(aEntries);

        assertEquals(aEntries.length, A.length());
    }

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        VectorOld B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new double[]{0, -924.34, 5, 1.34545};
        B = new VectorOld(bEntries);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameShape(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new double[]{0, -924.34, 5, 1.34545, 34.4};
        B = new VectorOld(bEntries);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameShape(B));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVectorOld B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(34, -0.34), new CNumber(0.445, 15.5), new CNumber(0.455), new CNumber(0, -8.435)};
        B = new CVectorOld(bEntries);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameShape(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(0.455), new CNumber(0, -8.435)};
        B = new CVectorOld(bEntries);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameShape(B));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new double[]{1.34545};
        indices = new int[]{2};
        size = 4;
        B = new CooVector(size, bEntries, indices);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameShape(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new double[]{1.45, 768.2, -1.55, 0.0234};
        indices = new int[]{1, 100, 502, 3405};
        size = 4096;
        B = new CooVector(size, bEntries, indices);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameShape(B));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(34.5, -9.632)};
        indices = new int[]{2};
        size = 4;
        B = new CooCVector(size, bEntries, indices);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameShape(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(34, -0.34), new CNumber(0.445, 15.5), new CNumber(0.455), new CNumber(0, -8.435)};
        indices = new int[]{1, 100, 502, 3405};
        size = 4096;
        B = new CooCVector(size, bEntries, indices);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameShape(B));
    }
}
