package com.flag4j.matrix;

import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixVectorCheckTests {
    double[][] aEntries;
    Matrix A;

    @Test
    void isVectorTestCase() {
        // ------------------ Sub-case 1 ------------------
        aEntries = new double[][]{{1.123, 5325.123}, {1.566, -2354.5767}};
        A = new Matrix(aEntries);
        assertFalse(A.isVector());

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}, {1.566, -2354.5767, 9}};
        A = new Matrix(aEntries);
        assertFalse(A.isVector());

        // ------------------ Sub-case 3 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}};
        A = new Matrix(aEntries);
        assertTrue(A.isVector());

        // ------------------ Sub-case 4 ------------------
        aEntries = new double[][]{{1.123}};
        A = new Matrix(aEntries);
        assertTrue(A.isVector());

        // ------------------ Sub-case 5 ------------------
        aEntries = new double[][]{{1.123}, {-9234}, {-7234.23}, {0.3242}, {1}};
        A = new Matrix(aEntries);
        assertTrue(A.isVector());
    }

    @Test
    void vectorTypeTestCase() {
        // ------------------ Sub-case 1 ------------------
        aEntries = new double[][]{{1.123, 5325.123}, {1.566, -2354.5767}};
        A = new Matrix(aEntries);
        assertEquals(-1, A.vectorType());

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}, {1.566, -2354.5767, 9}};
        A = new Matrix(aEntries);
        assertEquals(-1, A.vectorType());

        // ------------------ Sub-case 3 ------------------
        aEntries = new double[][]{{1.123, 5325.123, 123.4}};
        A = new Matrix(aEntries);
        assertEquals(1, A.vectorType());

        // ------------------ Sub-case 4 ------------------
        aEntries = new double[][]{{1.123}};
        A = new Matrix(aEntries);
        assertEquals(0, A.vectorType());

        // ------------------ Sub-case 5 ------------------
        aEntries = new double[][]{{1.123}, {-9234}, {-7234.23}, {0.3242}, {1}};
        A = new Matrix(aEntries);
        assertEquals(2, A.vectorType());
    }
}
