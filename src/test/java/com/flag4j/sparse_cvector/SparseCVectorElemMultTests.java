package com.flag4j.sparse_cvector;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseCVectorElemMultTests {

    SparseCVector a;
    int size;

    @Test
    void denseElemMultTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices;
        Vector b;
        CNumber[] expValues;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 0.14};
        b = new Vector(bValues);

        expValues = new CNumber[]{new CNumber(1.3345, -9.25).mult(1.223), new CNumber(0, -45.62).mult(3.4),
                new CNumber(25.612, 0.0245).mult(-14.51)};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void sparseElemMultTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices, bIndices;
        SparseVector b;
        CNumber[] expValues;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3};
        bIndices = new int[]{1, 2, 3, 5};
        b = new SparseVector(size, bValues, bIndices);

        expValues = new CNumber[]{aValues[1].mult(bValues[1]), aValues[2].mult(bValues[3])};
        expIndices = new int[]{2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        bIndices = new int[]{1, 2, 5, 6, 105, 132};
        b = new SparseVector(140, bValues, bIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void denseComplexElemMultTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] expIndices;
        CVector b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4),
                new CNumber(9924.515, 51.5), new CNumber(24.56, -88.351)};
        b = new CVector(bValues);

        expValues = new CNumber[]{
                new CNumber(1.3345, -9.25).mult(new CNumber(24.3, -0.013)),
                new CNumber(0, -45.62).mult(new CNumber(2.4)),
                new CNumber(25.612, 0.0245).mult(new CNumber(9924.515, 51.5))};
        expIndices = new int[]{0, 2, 5};
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
    void sparseComplexElemMultTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] bValues;
        int[] expIndices, bIndices;
        SparseCVector b;
        CNumber[] expValues;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{
                new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45)
        };
        bIndices = new int[]{1, 2, 3, 5};
        b = new SparseCVector(size, bValues, bIndices);

        expValues = new CNumber[]{aValues[1].mult(bValues[1]), aValues[2].mult(bValues[3])};
        expIndices = new int[]{2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{
                new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45),
                new CNumber(3249.56, 2122.2), new CNumber(-926.6, 324.67)
        };
        bIndices = new int[]{1, 2, 5, 6, 105, 132};
        b = new SparseCVector(140, bValues, bIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexScalarMultTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] expValues;
        int[] expIndices;
        CNumber b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        b = new CNumber(23.55, -984.2);

        expValues = new CNumber[]{
                new CNumber(1.3345, -9.25).mult(b),
                new CNumber(0, -45.62).mult(b),
                new CNumber(25.612, 0.0245).mult(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }


    @Test
    void realScalarMultTest() {
        CNumber[] aValues = {
                new CNumber(1.3345, -9.25), new CNumber(0, -45.62),
                new CNumber(25.612, 0.0245)};
        int[] aIndices = {0, 2, 5};
        size = 7;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] expValues;
        int[] expIndices;
        double b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        b = 24.5;

        expValues = new CNumber[]{
                new CNumber(1.3345, -9.25).mult(b),
                new CNumber(0, -45.62).mult(b),
                new CNumber(25.612, 0.0245).mult(b)};
        expIndices = new int[]{0, 2, 5};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }
}
