package com.flag4j.vector;

import com.flag4j.SparseVector;
import com.flag4j.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorAddSubEqTests {

    int[] indices;
    int size;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseAddEqTest() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2]};
        exp = new Vector(expEntries);

        A.addEq(B);

        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(IllegalArgumentException.class, () -> A.addEq(finalB));
    }


    @Test
    void realSparseAddEqTest() {
        double[] bEntries, expEntries;
        SparseVector B;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new SparseVector(size, bEntries, indices);
        expEntries = new double[]{1.34+34.677, 6.266, -90.45};
        exp = new Vector(expEntries);

        A.addEq(B);
        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new SparseVector(size, bEntries, indices);

        SparseVector finalB = B;
        assertThrows(IllegalArgumentException.class, () -> A.addEq(finalB));
    }


    @Test
    void doubleAddEqTest() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34+B, 6.266+B, -90.45+B};
        exp = new Vector(expEntries);

        A.addEq(B);
        assertEquals(exp, A);
    }

    // ----------------------------------------------------------------------------------------------


    @Test
    void realDenseSubTest() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]-bEntries[0], aEntries[1]-bEntries[1], aEntries[2]-bEntries[2]};
        exp = new Vector(expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(IllegalArgumentException.class, () -> A.subEq(finalB));
    }


    @Test
    void realSparseSubTest() {
        double[] bEntries, expEntries;
        SparseVector B;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new SparseVector(size, bEntries, indices);
        expEntries = new double[]{1.34-34.677, 6.266, -90.45};
        exp = new Vector(expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new SparseVector(size, bEntries, indices);

        SparseVector finalB = B;
        assertThrows(IllegalArgumentException.class, () -> A.subEq(finalB));
    }


    @Test
    void doubleSubTest() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34-B, 6.266-B, -90.45-B};
        exp = new Vector(expEntries);

        A.subEq(B);
        assertEquals(exp, A);
    }
}
