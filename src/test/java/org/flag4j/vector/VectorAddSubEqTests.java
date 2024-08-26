package org.flag4j.vector;

import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorAddSubEqTests {

    int[] indices;
    int size;

    double[] aEntries;
    VectorOld A;

    @Test
    void realDenseAddEqTestCase() {
        double[] bEntries, expEntries;
        VectorOld B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new VectorOld(bEntries);
        expEntries = new double[]{aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2]};
        exp = new VectorOld(expEntries);

        A.addEq(B);

        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new VectorOld(bEntries);

        VectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.addEq(finalB));
    }


    @Test
    void realSparseAddEqTestCase() {
        double[] bEntries, expEntries;
        CooVectorOld B;
        VectorOld exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new CooVectorOld(size, bEntries, indices);
        expEntries = new double[]{1.34+34.677, 6.266, -90.45};
        exp = new VectorOld(expEntries);

        A.addEq(B);
        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVectorOld(size, bEntries, indices);

        CooVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.addEq(finalB));
    }


    @Test
    void doubleAddEqTestCase() {
        double[] expEntries;
        double B = 1.5;
        VectorOld exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        expEntries = new double[]{1.34+B, 6.266+B, -90.45+B};
        exp = new VectorOld(expEntries);

        A.addEq(B);
        assertEquals(exp, A);
    }

    // ----------------------------------------------------------------------------------------------


    @Test
    void realDenseSubTestCase() {
        double[] bEntries, expEntries;
        VectorOld B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new VectorOld(bEntries);
        expEntries = new double[]{aEntries[0]-bEntries[0], aEntries[1]-bEntries[1], aEntries[2]-bEntries[2]};
        exp = new VectorOld(expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new VectorOld(bEntries);

        VectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.subEq(finalB));
    }


    @Test
    void realSparseSubTestCase() {
        double[] bEntries, expEntries;
        CooVectorOld B;
        VectorOld exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new CooVectorOld(size, bEntries, indices);
        expEntries = new double[]{1.34-34.677, 6.266, -90.45};
        exp = new VectorOld(expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVectorOld(size, bEntries, indices);

        CooVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.subEq(finalB));
    }


    @Test
    void doubleSubTestCase() {
        double[] expEntries;
        double B = 1.5;
        VectorOld exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new VectorOld(aEntries);
        expEntries = new double[]{1.34-B, 6.266-B, -90.45-B};
        exp = new VectorOld(expEntries);

        A.subEq(B);
        assertEquals(exp, A);
    }
}
