package com.flag4j.sparse_vector;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseVectorElemMultTests {

    SparseVector a;
    int size;

    @Test
    void sparseElemMultTest() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new SparseVector(size, aValues, aIndices);

        double[] bValues, expValues;
        int[] bIndices, expIndices;
        SparseVector b, exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, 44.51, 3.4};
        bIndices = new int[]{2, 81, 103};
        b = new SparseVector(size, bValues, bIndices);

        expValues = new double[]{1.223*1.34, 44.51*-0.00245};
        expIndices = new int[]{2, 81};
        exp = new SparseVector(151, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, 44.51, 3.4};
        bIndices = new int[]{2, 81, 103};
        b = new SparseVector(size-23, bValues, bIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void denseElemMultTest() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 3};
        size = 7;
        a = new SparseVector(size, aValues, aIndices);

        double[] bValues, expValues;
        int[] expIndices;
        Vector b;
        SparseVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 0.14};
        b = new Vector(bValues);

        expValues = new double[]{1.34*1.223, 51.6*3.4, -0.00245*2.3};
        expIndices = new int[]{0, 2, 3};
        exp = new SparseVector(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void sparseComplexElemMultTest() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new SparseVector(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] bIndices, expIndices;
        SparseCVector b, exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(2.441, -9.245), new CNumber(0, 4.51), new CNumber(24.5)};
        bIndices = new int[]{2, 81, 103};
        b = new SparseCVector(size, bValues, bIndices);

        expValues = new CNumber[]{
                new CNumber(2.441, -9.245).mult(1.34),
                new CNumber(0, 4.51).mult(-0.00245)};
        expIndices = new int[]{2, 81};
        exp = new SparseCVector(151, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(2.441, -9.245), new CNumber(0, 4.51), new CNumber(24.5)};
        bIndices = new int[]{2, 81, 103};
        b = new SparseCVector(size+134, bValues, bIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void denseComplexElemMultTest() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 3};
        size = 5;
        a = new SparseVector(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] expIndices;
        CVector b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4)};
        b = new CVector(bValues);

        expValues = new CNumber[]{new CNumber(24.3, -0.013).mult(1.34),
                new CNumber(2.4).mult(51.6), new CNumber(-994.1 ,1.45).mult(-0.00245)};
        expIndices = new int[]{0, 2, 3};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6), new CNumber(24),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void doubleScalarElemMultTest() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new SparseVector(size, aValues, aIndices);

        double b;

        double[] expValues;
        int[] expIndices;
        SparseVector exp;

        // -------------------- Sub-case 1 --------------------
        b = 24.56;

        expValues = new double[]{1.34*b, 51.6*b, -0.00245*b, 99.2456*b, -1005.6*b};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new SparseVector(151, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }


    @Test
    void complexScalarElemMultTest() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new SparseVector(size, aValues, aIndices);

        CNumber b;

        CNumber[] expValues;
        int[] expIndices;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        b = new CNumber(234.6677, -9.35);

        expValues = new CNumber[]{b.mult(1.34), b.mult(51.6), b.mult(-0.00245), b.mult(99.2456), b.mult(-1005.6)};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new SparseCVector(151, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }
}
