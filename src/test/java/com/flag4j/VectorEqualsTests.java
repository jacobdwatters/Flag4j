package com.flag4j;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorEqualsTests {

    int[] indices;
    int sparseSize;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseTest() {
        double[] bEntries;
        Vector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 543.354, -0.3456};
        B = new Vector(bEntries);

        assertTrue(A.equals(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 123.5, -0.3456};
        B = new Vector(bEntries);

        assertFalse(A.equals(B));


        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 543.354, -0.3456, 0};
        B = new Vector(bEntries);

        assertFalse(A.equals(B));
    }


    @Test
    void complexDenseTest() {
        CNumber[] bEntries;
        CVector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(1.234), new CNumber(543.354), new CNumber(-0.3456)};
        B = new CVector(bEntries);

        assertTrue(A.equals(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(1.234, 13.4), new CNumber(543.354), new CNumber(-0.3456)};
        B = new CVector(bEntries);

        assertFalse(A.equals(B));


        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(1.234), new CNumber(543.354), new CNumber(-0.3456), new CNumber(0, 123.5)};
        B = new CVector(bEntries);

        assertFalse(A.equals(B));
    }


    @Test
    void realSparseTest() {
        double[] bEntries;
        SparseVector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{1};
        sparseSize = 3;
        B = new SparseVector(sparseSize, bEntries, indices);

        assertTrue(A.equals(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{2};
        sparseSize = 3;
        B = new SparseVector(sparseSize, bEntries, indices);

        assertFalse(A.equals(B));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{1};
        sparseSize = 4;
        B = new SparseVector(sparseSize, bEntries, indices);

        assertFalse(A.equals(B));
    }


    @Test
    void complexSparseTest() {
        CNumber[] bEntries;
        SparseCVector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(543.354)};
        indices = new int[]{1};
        sparseSize = 3;
        B = new SparseCVector(sparseSize, bEntries, indices);

        assertTrue(A.equals(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(543.354, -9.34)};
        indices = new int[]{1};
        sparseSize = 3;
        B = new SparseCVector(sparseSize, bEntries, indices);

        assertFalse(A.equals(B));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(543.354)};
        indices = new int[]{1};
        sparseSize = 4;
        B = new SparseCVector(sparseSize, bEntries, indices);

        assertFalse(A.equals(B));
    }


    @Test
    void objectTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{123.4};
        A = new Vector(aEntries);
        Double B = 123.4;

        assertFalse(A.equals(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        String BString = "hello";

        assertFalse(A.equals(BString));
    }
}
