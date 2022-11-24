package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixAgglomerationTests {

    Matrix A;
    double[][] aEntries;
    double exp, act;
    int[] expInts, actInts;

    @Test
    void sumTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = 45;

        act = A.sum();

        assertEquals(exp, act);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        exp = -568.191616;

        act = A.sum();

        assertEquals(exp, act);
    }


    @Test
    void minTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = 1;

        act = A.min();

        assertEquals(exp, act);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        exp = -924;

        act = A.min();

        assertEquals(exp, act);
    }


    @Test
    void minAbsTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = 1;

        act = A.minAbs();

        assertEquals(exp, act);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        exp = 0.123;

        act = A.minAbs();

        assertEquals(exp, act);
    }


    @Test
    void minArgTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        expInts = new int[]{0, 0};

        actInts = A.argMin();

        assertArrayEquals(expInts, actInts);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        expInts = new int[]{3, 1};

        actInts = A.argMin();

        assertArrayEquals(expInts, actInts);
    }


    @Test
    void maxTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = 9;

        act = A.max();

        assertEquals(exp, act);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        exp = 234.656;

        act = A.max();

        assertEquals(exp, act);
    }


    @Test
    void maxAbsTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = 9;

        act = A.maxAbs();

        assertEquals(exp, act);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        exp = 924;

        act = A.maxAbs();

        assertEquals(exp, act);
    }

    @Test
    void maxArgTest() {
        // ------------ Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        expInts = new int[]{2, 2};

        actInts = A.argMax();

        assertArrayEquals(expInts, actInts);

        // ------------ Sub-case 2 ---------------
        aEntries = new double[][]{{-0.123, 234.656}, {0.932344, -85.2}, {103.43, 9.11304}, {93, -924}};
        A = new Matrix(aEntries);
        expInts = new int[]{0, 1};

        actInts = A.argMax();

        assertArrayEquals(expInts, actInts);
    }
}
