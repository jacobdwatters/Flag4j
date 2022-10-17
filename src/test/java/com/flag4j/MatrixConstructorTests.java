package com.flag4j;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixConstructorTests {
    double[][] expEntriesA;
    int expNumColsA, expNumRowsA;
    Matrix A;


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
        fillValue = -1231.012;
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
}
