package com.flag4j.complex_vector;

import com.flag4j.CVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CVectorAddTests {

    CNumber[] aEntries = new CNumber[]{
        new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
        new CNumber(3.54698), new CNumber(0, 8.356)};
    CVector a = new CVector(aEntries);
    CNumber[] expEntries;
    CVector exp;

    int sparseSize;
    int[] sparseIndices;

    @Test
    void scalDoubleTest() {
        double b;

        // ------------------ Sub-case 1 ------------------
        b = 45.15;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        b = -2384.526;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 3 ------------------
        b = Double.POSITIVE_INFINITY;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 4 ------------------
        b = Double.NaN;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        CVector act = a.add(b);

        for(int i=0; i<act.size; i++) {
            assertTrue(Double.isNaN(act.get(i).re));
            assertEquals(exp.get(i).im, act.get(i).im);
        }
    }


    @Test
    void scalCNumberTest() {
        CNumber b;

        // ------------------ Sub-case 1 ------------------
        b = new CNumber(9.145, -523923.15965);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        b = new CNumber(0, -14.36);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 3 ------------------
        b = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 4 ------------------
        b = new CNumber(6.24, Double.POSITIVE_INFINITY);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 5 ------------------
        b = new CNumber(Double.NEGATIVE_INFINITY, 135.5);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 6 ------------------
        b = new CNumber(Double.NaN, Double.NaN);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        CVector act = a.add(b);

        for (int i = 0; i < act.size; i++) {
            assertTrue(Double.isNaN(act.get(i).re));
            assertTrue(Double.isNaN(act.get(i).im));
        }
    }


    @Test
    void realDenseTest() {
        double[] bEntries;
        Vector b;

        // ------------------ Sub-case 1 ------------------
        bEntries = new double[]{54.1354, -99.2344, 0, 0.023};
        b = new Vector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        bEntries = new double[]{-54.1354, -99.2344, 0, Double.NEGATIVE_INFINITY};
        b = new Vector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 3 ------------------
        bEntries = new double[]{-54.1354, -99.2344, 0, 14, 1.5};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));

        // ------------------ Sub-case 4 ------------------
        bEntries = new double[]{-54.1354, -99.2344};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB2));
    }


    @Test
    void realSparseTest() {
        double[] bEntries;
        SparseVector b;

        // ------------------ Sub-case 1 ------------------
        bEntries = new double[]{54.1354, -1.4};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(54.1354), new CNumber(-24.565, 9.3),
                new CNumber(3.54698).add(-1.4), new CNumber(0, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356).add(-1.4)};
        exp = new CVector(expEntries);

        CVector act = a.add(b);

        System.out.print("\nexp: ");
        for(int i=0; i<exp.size; i++) {
            System.out.print(exp.get(i) + "    ");
        }

        System.out.print("\nact: ");
        for(int i=0; i<act.size; i++) {
            System.out.print(act.get(i) + "    ");
        }

        assertEquals(exp, act);

//        // ------------------ Sub-case 3 ------------------
//        bEntries = new double[]{-1.4};
//        sparseSize = 4234;
//        sparseIndices = new int[]{3};
//        b = new SparseVector(sparseSize, bEntries, sparseIndices);
//
//        SparseVector finalB = b;
//        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));
//
//        // ------------------ Sub-case 4 ------------------
//        bEntries = new double[]{-1.4};
//        sparseSize = 3;
//        sparseIndices = new int[]{3};
//        b = new SparseVector(sparseSize, bEntries, sparseIndices);
//
//        SparseVector finalB2 = b;
//        assertThrows(IllegalArgumentException.class, ()->a.add(finalB2));
    }


    @Test
    void complexDenseTest() {
        CNumber[] bEntries;
        CVector b;

        // ------------------ Sub-case 1 ------------------
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35), new CNumber(-9924.5, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        bEntries = new CNumber[]{new CNumber(2455, 0.0001424), new CNumber(-9),
                new CNumber(-0.0, Double.NEGATIVE_INFINITY), new CNumber(Double.POSITIVE_INFINITY, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 3 ------------------
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35), new CNumber(-9924.5, 24.656),
                new CNumber(9.345, 1344)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB));

        // ------------------ Sub-case 4 ------------------
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.add(finalB2));
    }
}
