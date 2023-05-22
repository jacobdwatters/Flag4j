package com.flag4j.sparse_vector;


import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseVectorAddTests {
    SparseVector a;

    @Test
    void sparseAddTest() {
        SparseVector b, exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new SparseVector(size, aValues, aIndices);

        double[] bValues = {44, -5.66, 22.445, -0.994, 10.5};
        int[] bIndices = {1, 5, 11, 67, 200};
        b = new SparseVector(size, bValues, bIndices);

        // --------------------- Sub-case 1 ---------------------
        double[] expValues = {1.34, 44, 51.6-5.66, 22.445, -0.994, -0.00245, 10.5};
        int[] expIndices = {0, 1, 5, 11, 67, 103, 200};
        exp = new SparseVector(size, expValues, expIndices);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new double[]{44, -5.66, 22.445, -0.994, 10.5};
        bIndices = new int[]{1, 5, 11, 67, 200};
        b = new SparseVector(size+13, bValues, bIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));
    }


    @Test
    void sparseComplexAddTest() {
        SparseCVector b, exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new SparseVector(size, aValues, aIndices);

        CNumber[] bValues = {new CNumber(1, -0.024),
                new CNumber(99.24, 1.5), new CNumber(0, 1.4)};
        int[] bIndices = {1, 5, 6};
        b = new SparseCVector(size, bValues, bIndices);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(1.34), new CNumber(1, -0.024),
                new CNumber(99.24+51.6, 1.5), new CNumber(0, 1.4), new CNumber(-0.00245)};
        int[] expIndices = {0, 1, 5, 6, 103};
        exp = new SparseCVector(size, expValues, expIndices);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new CNumber[]{new CNumber(1, -0.024),
                new CNumber(99.24, 1.5), new CNumber(0, 1.4)};
        bIndices = new int[]{1, 5, 6};
        b = new SparseCVector(size+13, bValues, bIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));
    }


    @Test
    void denseTest() {
        Vector b, exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new SparseVector(size, aValues, aIndices);

        double[] bValues = {1, 5, -0.0024, 1, 2001.256, 61, -99.24, 1.5};
        b = new Vector(bValues);

        // --------------------- Sub-case 1 ---------------------
        double[] expValues = {1+1.34, 5, -0.0024+51.6, 1, 2001.256, 61-0.00245, -99.24, 1.5};
        exp = new Vector(expValues);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new double[]{1, 5, -0.0024, 1, 2001.256, 61};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));
    }


    @Test
    void denseComplexTest() {
        CVector b, exp;

        double[] aValues = {1.34, 51.6};
        int[] aIndices = {0, 2};
        int size = 5;
        a = new SparseVector(size, aValues, aIndices);

        CNumber[] bValues = {new CNumber(1.445, -9.24), new CNumber(1.45),
        new CNumber(0, -99.145), new CNumber(4.51, 8.456), new CNumber(11.34, -0.00245)};
        b = new CVector(bValues);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(1.445+1.34, -9.24), new CNumber(1.45),
                new CNumber(51.6, -99.145), new CNumber(4.51, 8.456), new CNumber(11.34, -0.00245)};
        exp = new CVector(expValues);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new CNumber[]{new CNumber(1.445, -9.24), new CNumber(1.45),
                new CNumber(0, -99.145), new CNumber(4.51, 8.456),
                new CNumber(11.34, -0.00245), new CNumber(34.5, 0.0014)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));
    }


    @Test
    void scalarTest() {
        double b;
        Vector exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new SparseVector(size, aValues, aIndices);

        b = 2.345;

        // --------------------- Sub-case 1 ---------------------
        double[] expValues = {1.34+2.345, 2.345, 51.6+2.345, 2.345, 2.345, -0.00245+2.345, 2.345, 2.345};
        exp = new Vector(expValues);

        assertEquals(exp, a.add(b));
    }


    @Test
    void complexScalarTest() {
        CNumber b;
        CVector exp;

        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 3};
        int size = 5;
        a = new SparseVector(size, aValues, aIndices);

        b = new CNumber(13.455, -1459.4521);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(13.455+1.34, -1459.4521), new CNumber(13.455, -1459.4521),
                new CNumber(13.455+51.6, -1459.4521), new CNumber(13.455-0.00245, -1459.4521),
                new CNumber(13.455, -1459.4521)};
        exp = new CVector(expValues);

        assertEquals(exp, a.add(b));
    }
}
