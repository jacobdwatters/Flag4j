package com.flag4j.matrix.vector;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class VectorShapeTests {

    int[] indices;
    int size;

    double[] aEntries;
    Vector A;


    @Test
    void realDenseTest() {
        double[] bEntries;
        Vector B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{0, -924.34, 5, 1.34545};
        B = new Vector(bEntries);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameSize(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{0, -924.34, 5, 1.34545, 34.4};
        B = new Vector(bEntries);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameSize(B));
    }


    @Test
    void complexDenseTest() {
        CNumber[] bEntries;
        CVector B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34, -0.34), new CNumber(0.445, 15.5), new CNumber(0.455), new CNumber(0, -8.435)};
        B = new CVector(bEntries);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameSize(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(0.455), new CNumber(0, -8.435)};
        B = new CVector(bEntries);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameSize(B));
    }


    @Test
    void realSparseTest() {
        double[] bEntries;
        SparseVector B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{1.34545};
        indices = new int[]{2};
        size = 4;
        B = new SparseVector(size, bEntries, indices);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameSize(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new double[]{1.45, 768.2, -1.55, 0.0234};
        indices = new int[]{1, 100, 502, 3405};
        size = 4096;
        B = new SparseVector(size, bEntries, indices);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameSize(B));
    }


    @Test
    void complexSparseTest() {
        CNumber[] bEntries;
        SparseCVector B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34.5, -9.632)};
        indices = new int[]{2};
        size = 4;
        B = new SparseCVector(size, bEntries, indices);

        assertTrue(A.sameShape(B));
        assertTrue(A.sameSize(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[]{1.23, 45, -0.435, 22.15};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34, -0.34), new CNumber(0.445, 15.5), new CNumber(0.455), new CNumber(0, -8.435)};
        indices = new int[]{1, 100, 502, 3405};
        size = 4096;
        B = new SparseCVector(size, bEntries, indices);

        assertFalse(A.sameShape(B));
        assertFalse(A.sameSize(B));
    }
}
