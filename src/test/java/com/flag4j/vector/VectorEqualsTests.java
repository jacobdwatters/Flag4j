package com.flag4j.vector;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CVector;
import com.flag4j.dense.Vector;
import com.flag4j.sparse.CooCVector;
import com.flag4j.sparse.CooVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorEqualsTests {

    int[] indices;
    int sparseSize;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 543.354, -0.3456};
        B = new Vector(bEntries);

        assertEquals(A, B);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 123.5, -0.3456};
        B = new Vector(bEntries);

        assertNotEquals(A, B);


        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new double[]{1.234, 543.354, -0.3456, 0};
        B = new Vector(bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(1.234), new CNumber(543.354), new CNumber(-0.3456)};
        B = new CVector(bEntries);

        assertEquals(A, B);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(1.234, 13.4), new CNumber(543.354), new CNumber(-0.3456)};
        B = new CVector(bEntries);

        assertNotEquals(A, B);


        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{1.234, 543.354, -0.3456};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(1.234), new CNumber(543.354), new CNumber(-0.3456), new CNumber(0, 123.5)};
        B = new CVector(bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{1};
        sparseSize = 3;
        B = new CooVector(sparseSize, bEntries, indices);

        assertEquals(A, B);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{2};
        sparseSize = 3;
        B = new CooVector(sparseSize, bEntries, indices);

        assertNotEquals(A, B);

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new double[]{543.354};
        indices = new int[]{1};
        sparseSize = 4;
        B = new CooVector(sparseSize, bEntries, indices);

        assertNotEquals(A, B);
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector B;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(543.354)};
        indices = new int[]{1};
        sparseSize = 3;
        B = new CooCVector(sparseSize, bEntries, indices);

        assertEquals(A, B);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(543.354, -9.34)};
        indices = new int[]{1};
        sparseSize = 3;
        B = new CooCVector(sparseSize, bEntries, indices);

        assertNotEquals(A, B);

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(543.354)};
        indices = new int[]{1};
        sparseSize = 4;
        B = new CooCVector(sparseSize, bEntries, indices);

        assertNotEquals(A, B);
    }


    @Test
    void objectTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{123.4};
        A = new Vector(aEntries);
        Double B = 123.4;

        assertNotEquals(A, B, 0.0);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0, 543.354, 0};
        A = new Vector(aEntries);
        String BString = "hello";

        assertNotEquals(A, BString);
    }
}
