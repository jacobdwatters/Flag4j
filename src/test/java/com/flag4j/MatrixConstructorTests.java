package com.flag4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MatrixConstructorTests {
    double[][] expEntriesA;
    int expNumColsA, expNumRowsA;
    Matrix A, B, C;


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

}
