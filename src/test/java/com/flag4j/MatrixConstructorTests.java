package com.flag4j;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
class MatrixConstructorTests {
    double[][] expEntriesA;
    int[][] expEntriesAint;
    int expNumColsA, expNumRowsA;
    Matrix A, B;

    /**
     * Tests the default constructor for the Matrix class.
     */
    @Test
    void defaultConstructorTestCase() {
        expEntriesA = new double[][]{};
        expNumColsA = expNumRowsA = 0;

        A = new Matrix();

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);
    }

    /**
     * Tests size constructor for the matrix class.
     */
    @Test
    void sizeConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        int size = 0;
        expEntriesA = new double[size][size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 2 ------------
        size = 1;
        expEntriesA = new double[size][size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 3 ------------
        size = 145;
        expEntriesA = new double[size][size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 4 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-1));

        // ------------ Sub-case 5 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-5));
    }


    /**
     * Tests size constructor for the matrix class.
     */
    @Test
    void sizeFillConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        int size = 0;
        double fillValue = -1231.012;
        expEntriesA = new double[size][size];
        double[] row = new double[size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 2 ------------
        size = 15;
        fillValue = 13231.12321;
        expEntriesA = new double[size][size];
        row = new double[size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 3 ------------
        size = 145;
        fillValue = 0;
        expEntriesA = new double[size][size];
        row = new double[size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 4 ------------
        size = 13;
        fillValue = -1231;
        expEntriesA = new double[size][size];
        row = new double[size];
        expNumColsA = expNumRowsA = size;

        A = new Matrix(size, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 5 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-1));

        // ------------ Sub-case 6 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-5));
    }


    /**
     * Tests constructor which takes the number of rows and columns.
     */
    @Test
    void rowColConstructorTestCase() {
        // ------------- Sub-case 1 ------------------
        expNumRowsA = 0;
        expNumColsA = 0;
        expEntriesA = new double[expNumRowsA][expNumColsA];

        A = new Matrix(expNumRowsA, expNumRowsA);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------- Sub-case 2 ------------------
        expNumRowsA = 1;
        expNumColsA = 0;
        expEntriesA = new double[expNumRowsA][expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------- Sub-case 3 ------------------
        expNumRowsA = 0;
        expNumColsA = 1;
        expEntriesA = new double[expNumRowsA][expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------- Sub-case 4 ------------------
        expNumRowsA = 12;
        expNumColsA = 51;
        expEntriesA = new double[expNumRowsA][expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------- Sub-case 5 ------------------
        expNumRowsA = 15;
        expNumColsA = 1;
        expEntriesA = new double[expNumRowsA][expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);


        // ------------ Sub-case 6 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, 2));

        // ------------ Sub-case 7 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(2, -2));

        // ------------ Sub-case 8 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, -2));
    }


    @Test
    void rowColFillConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expNumRowsA = 0;
        expNumColsA = 1;
        double fillValue = -1231.012;
        expEntriesA = new double[expNumRowsA][expNumColsA];
        double[] row = new double[expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 2 ------------
        expNumRowsA = 1;
        expNumColsA = 0;
        fillValue = 13231.12321;
        expEntriesA = new double[expNumRowsA][expNumColsA];
        row = new double[expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 3 ------------
        expNumRowsA = 12;
        expNumColsA = 14;
        fillValue = 0;
        expEntriesA = new double[expNumRowsA][expNumColsA];
        row = new double[expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 4 ------------
        expNumRowsA = 1;
        expNumColsA = 19;
        fillValue = -144;
        expEntriesA = new double[expNumRowsA][expNumColsA];
        row = new double[expNumColsA];

        A = new Matrix(expNumRowsA, expNumColsA, fillValue);
        Arrays.fill(row, fillValue);
        Arrays.fill(expEntriesA, row);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);

        // ------------ Sub-case 5 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, 2, 45));

        // ------------ Sub-case 6 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(2, -2, 45));

        // ------------ Sub-case 7 ------------
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-2, -2, 45));
    }


    @Test
    void doubleEntriesConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expEntriesA = new double[][]
                {{1, 5.2234, 1231.2344, -112, 3.3, 0, 2.3e-4},
                {0.4, 12, -44.3, Double.NEGATIVE_INFINITY, Math.PI, Double.NaN, Double.POSITIVE_INFINITY}};
        expNumRowsA = expEntriesA.length;
        expNumColsA = expEntriesA[0].length;
        double[] row = new double[expNumColsA];

        A = new Matrix(expEntriesA);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);
    }


    @Test
    void intEntriesConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expEntriesA = new double[][]
                {{1, 2, 3},
                {0, -1, -3},
                {Integer.MAX_VALUE, Integer.MIN_VALUE, 00341}};
        expEntriesAint = new int[][]
                {{1, 2, 3},
                {0, -1, -3},
                {Integer.MAX_VALUE, Integer.MIN_VALUE, 00341}};
        expNumRowsA = expEntriesAint.length;
        expNumColsA = expEntriesAint[0].length;
        double[] row = new double[expNumColsA];

        A = new Matrix(expEntriesAint);

        assertArrayEquals(expEntriesA, A.entries);
        assertEquals(expNumRowsA, A.m);
        assertEquals(expNumColsA, A.n);
    }


    @Test void matConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expEntriesA = new double[][]
                {{1, 5.2234, 1231.2344, -112, 3.3, 0, 2.3e-4},
                        {0.4, 12, -44.3, Double.NEGATIVE_INFINITY, Math.PI, Double.NaN, Double.POSITIVE_INFINITY}};
        expNumRowsA = expEntriesA.length;
        expNumColsA = expEntriesA[0].length;
        double[] row = new double[expNumColsA];

        A = new Matrix(expEntriesA);
        B = new Matrix(A);

        assertArrayEquals(A.entries, B.entries);
        assertEquals(A.m, B.m);
        assertEquals(A.n, B.n);
    }
}
