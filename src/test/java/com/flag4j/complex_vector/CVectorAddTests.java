package com.flag4j.complex_vector;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CVector;
import com.flag4j.dense.Vector;
import com.flag4j.exceptions.LinearAlgebraException;
import com.flag4j.sparse.CooCVector;
import com.flag4j.sparse.CooVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CVectorAddTests {

    CNumber[] aEntries;
    CVector a;
    CNumber[] expEntries;
    CVector exp;

    int sparseSize;
    int[] sparseIndices;


    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356)};
        a = new CVector(aEntries);
    }


    @Test
    void scalDoubleTestCase() {
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
    void scalCNumberTestCase() {
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
    void realDenseTestCase() {
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
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));

        // ------------------ Sub-case 4 ------------------
        bEntries = new double[]{-54.1354, -99.2344};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB2));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;

        // ------------------ Sub-case 1 ------------------
        bEntries = new double[]{54.1354, -1.4};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(54.1354), new CNumber(-24.565, 9.3),
                new CNumber(3.54698).add(-1.4), new CNumber(0, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356).add(-1.4)};
        exp = new CVector(expEntries);

        CVector act = a.add(b);
        assertEquals(exp, act);

        // ------------------ Sub-case 3 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));

        // ------------------ Sub-case 4 ------------------
        bEntries = new double[]{-1.4};
        sparseSize = 3;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB2));
    }


    @Test
    void complexDenseTestCase() {
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
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));

        // ------------------ Sub-case 4 ------------------
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB2));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector b;

        // ------------------ Sub-case 1 ------------------
        bEntries = new CNumber[]{new CNumber(-9.24, 8.14), new CNumber(0, 22455.6126)};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(bEntries[0]), new CNumber(-24.565, 9.3),
                new CNumber(3.54698).add(bEntries[1]), new CNumber(0, 8.356)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.add(b));

        // ------------------ Sub-case 2 ------------------
        bEntries = new CNumber[]{new CNumber(4.5, 0.00245)};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356).add(bEntries[0])};
        exp = new CVector(expEntries);

        CVector act = a.add(b);
        assertEquals(exp, act);

        // ------------------ Sub-case 3 ------------------
        bEntries = new CNumber[]{new CNumber(9.3455, 15.6)};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));

        // ------------------ Sub-case 4 ------------------
        bEntries = new CNumber[]{new CNumber(9.3455, 15.6)};
        sparseSize = 3;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB2));
    }

    // ----------- ADD EQ TESTS -----------

    @Test
    void scalDoubleEqTestCase() {
        double b;

        // ------------------ Sub-case 1 ------------------
        setup();
        b = 45.15;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        setup();
        b = -2384.526;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        setup();
        b = Double.POSITIVE_INFINITY;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 4 ------------------
        setup();
        b = Double.NaN;
        expEntries = new CNumber[]{
                new CNumber(2.566+b, -9.24), new CNumber(-24.565+b, 9.3),
                new CNumber(3.54698+b), new CNumber(0+b, 8.356)};
        exp = new CVector(expEntries);

        a.addEq(b);

        for(int i=0; i<a.size; i++) {
            assertTrue(Double.isNaN(a.get(i).re));
            assertEquals(exp.get(i).im, a.get(i).im);
        }
    }


    @Test
    void scalCNumberEqTestCase() {
        CNumber b;

        // ------------------ Sub-case 1 ------------------
        setup();
        b = new CNumber(9.145, -523923.15965);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        setup();
        b = new CNumber(0, -14.36);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        setup();
        b = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 4 ------------------
        setup();
        b = new CNumber(6.24, Double.POSITIVE_INFINITY);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 5 ------------------
        setup();
        b = new CNumber(Double.NEGATIVE_INFINITY, 135.5);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 6 ------------------
        setup();
        b = new CNumber(Double.NaN, Double.NaN);
        expEntries = new CNumber[]{
                new CNumber(2.566, -9.24).add(b), new CNumber(-24.565, 9.3).add(b),
                new CNumber(3.54698).add(b), new CNumber(0, 8.356).add(b)};
        exp = new CVector(expEntries);

        a.addEq(b);

        for (int i = 0; i < a.size; i++) {
            assertTrue(Double.isNaN(a.get(i).re));
            assertTrue(Double.isNaN(a.get(i).im));
        }
    }


    @Test
    void realDenseEqTestCase() {
        double[] bEntries;
        Vector b;

        // ------------------ Sub-case 1 ------------------
        setup();
        bEntries = new double[]{54.1354, -99.2344, 0, 0.023};
        b = new Vector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        setup();
        bEntries = new double[]{-54.1354, -99.2344, 0, Double.NEGATIVE_INFINITY};
        b = new Vector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        setup();
        bEntries = new double[]{-54.1354, -99.2344, 0, 14, 1.5};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB));

        // ------------------ Sub-case 4 ------------------
        setup();
        bEntries = new double[]{-54.1354, -99.2344};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB2));
    }


    @Test
    void realSparseEqTestCase() {
        double[] bEntries;
        CooVector b;

        // ------------------ Sub-case 1 ------------------
        setup();
        bEntries = new double[]{54.1354, -1.4};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(54.1354), new CNumber(-24.565, 9.3),
                new CNumber(3.54698).add(-1.4), new CNumber(0, 8.356)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        setup();
        bEntries = new double[]{-1.4};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356).add(-1.4)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        setup();
        bEntries = new double[]{-1.4};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB));

        // ------------------ Sub-case 4 ------------------
        setup();
        bEntries = new double[]{-1.4};
        sparseSize = 3;
        sparseIndices = new int[]{3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB2));
    }


    @Test
    void complexDenseEqTestCase() {
        CNumber[] bEntries;
        CVector b;

        // ------------------ Sub-case 1 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35), new CNumber(-9924.5, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(2455, 0.0001424), new CNumber(-9),
                new CNumber(-0.0, Double.NEGATIVE_INFINITY), new CNumber(Double.POSITIVE_INFINITY, 24.656)};
        b = new CVector(bEntries);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(b.get(0)), new CNumber(-24.565, 9.3).add(b.get(1)),
                new CNumber(3.54698).add(b.get(2)), new CNumber(0, 8.356).add(b.get(3))};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35), new CNumber(-9924.5, 24.656),
                new CNumber(9.345, 1344)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB));

        // ------------------ Sub-case 4 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(2.45, -99.24), new CNumber(9),
                new CNumber(0, -8.35)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB2));
    }


    @Test
    void complexSparseEqTestCase() {
        CNumber[] bEntries;
        CooCVector b;

        // ------------------ Sub-case 1 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(-9.24, 8.14), new CNumber(0, 22455.6126)};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24).add(bEntries[0]), new CNumber(-24.565, 9.3),
                new CNumber(3.54698).add(bEntries[1]), new CNumber(0, 8.356)};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 2 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(4.5, 0.00245)};
        sparseSize = 4;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries  = new CNumber[]{
                new CNumber(2.566, -9.24), new CNumber(-24.565, 9.3),
                new CNumber(3.54698), new CNumber(0, 8.356).add(bEntries[0])};
        exp = new CVector(expEntries);

        a.addEq(b);
        assertEquals(exp, a);

        // ------------------ Sub-case 3 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(9.3455, 15.6)};
        sparseSize = 4234;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB));

        // ------------------ Sub-case 4 ------------------
        setup();
        bEntries = new CNumber[]{new CNumber(9.3455, 15.6)};
        sparseSize = 3;
        sparseIndices = new int[]{3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB2 = b;
        assertThrows(LinearAlgebraException.class, ()->a.addEq(finalB2));
    }
}
