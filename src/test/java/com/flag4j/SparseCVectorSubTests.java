package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseCVectorSubTests {
    SparseCVector a;

    @Test
    void sparseSubTest() {
        SparseVector b;
        SparseCVector exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new SparseCVector(size, aValues, aIndices);

        double[] bValues = {44, -5.66, 22.445, -0.994, 10.5};
        int[] bIndices = {1, 5, 11, 67, 200};
        b = new SparseVector(size, bValues, bIndices);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98), new CNumber(44).addInv(), new CNumber(-8.2, 55.1).sub(-5.66),
                new CNumber(22.445).addInv(), new CNumber(-0.994).addInv(), new CNumber(0, 14.5), new CNumber(10.50).addInv()};
        int[] expIndices = {0, 1, 5, 11, 67, 103, 200};
        exp = new SparseCVector(size, expValues, expIndices);

        assertEquals(exp, a.sub(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new double[]{44, -5.66, 22.445, -0.994, 10.5};
        bIndices = new int[]{1, 5, 11, 67, 200};
        b = new SparseVector(size+13, bValues, bIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.sub(finalB));
    }


    @Test
    void sparseComplexSubTest() {
        SparseCVector b, exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] bValues = {new CNumber(1, -0.024),
                new CNumber(99.24, 1.5), new CNumber(0, 1.4)};
        int[] bIndices = {1, 5, 6};
        b = new SparseCVector(size, bValues, bIndices);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98), new CNumber(1, -0.024).addInv(),
                new CNumber(-8.2, 55.1).sub(new CNumber(99.24, 1.5)), new CNumber(0, 1.4).addInv(),
                new CNumber(0, 14.5)
        };
        int[] expIndices = {0, 1, 5, 6, 103};
        exp = new SparseCVector(size, expValues, expIndices);

        assertEquals(exp, a.sub(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new CNumber[]{new CNumber(1, -0.024),
                new CNumber(99.24, 1.5), new CNumber(0, 1.4)};
        bIndices = new int[]{1, 5, 6};
        b = new SparseCVector(size+13, bValues, bIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.sub(finalB));
    }


    @Test
    void denseTest() {
        Vector b;
        CVector exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new SparseCVector(size, aValues, aIndices);

        double[] bValues = {1, 5, -0.0024, 1, 2001.256, 61, -99.24, 1.5};
        b = new Vector(bValues);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {
                aValues[0].sub(new CNumber(1)), new CNumber(5).addInv(),
                aValues[1].sub(new CNumber(-0.0024)), new CNumber(1).addInv(),
                new CNumber(2001.256).addInv(), aValues[2].sub(new CNumber(61)),
                new CNumber(-99.24).addInv(), new CNumber(1.5).addInv()};
        exp = new CVector(expValues);

        assertEquals(exp, a.sub(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new double[]{1, 5, -0.0024, 1, 2001.256, 61};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.sub(finalB));
    }


    @Test
    void denseComplexTest() {
        CVector b, exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1)};
        int[] aIndices = {0, 2};
        int size = 5;
        a = new SparseCVector(size, aValues, aIndices);

        CNumber[] bValues = {new CNumber(1.445, -9.24), new CNumber(1.45),
                new CNumber(0, -99.145), new CNumber(4.51, 8.456), new CNumber(11.34, -0.00245)};
        b = new CVector(bValues);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98).sub(new CNumber(1.445, -9.24)), new CNumber(1.45).addInv(),
                new CNumber(-8.2, 55.1).sub(new CNumber(0, -99.145)), new CNumber(4.51, 8.456).addInv(),
                new CNumber(11.34, -0.00245).addInv()};
        exp = new CVector(expValues);

        assertEquals(exp, a.sub(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new CNumber[]{new CNumber(1.445, -9.24), new CNumber(1.45),
                new CNumber(0, -99.145), new CNumber(4.51, 8.456),
                new CNumber(11.34, -0.00245), new CNumber(34.5, 0.0014)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.sub(finalB));
    }


    @Test
    void scalarTest() {
        double b;
        CVector exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new SparseCVector(size, aValues, aIndices);

        b = 2.345;

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {aValues[0].sub(new CNumber(b)), new CNumber(-b), aValues[1].sub(new CNumber(b)), new CNumber(-b),
                new CNumber(-b), aValues[2].sub(new CNumber(b)), new CNumber(-b), new CNumber(-b)};
        exp = new CVector(expValues);

        assertEquals(exp, a.sub(b));
    }


    @Test
    void complexScalarTest() {
        CNumber b;
        CVector exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 2, 3};
        int size = 5;
        a = new SparseCVector(size, aValues, aIndices);

        b = new CNumber(13.455, -1459.4521);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98).sub(b), b.addInv(), new CNumber(-8.2, 55.1).sub(b),
                new CNumber(0, 14.5).sub(b), b.addInv()};
        exp = new CVector(expValues);

        assertEquals(exp, a.sub(b));
    }
}
