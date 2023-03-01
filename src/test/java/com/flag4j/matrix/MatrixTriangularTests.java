package com.flag4j.matrix;

import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixTriangularTests {
    double[][] aEntries;
    Matrix A;

    @Test
    void triangularTests() {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{{1, 3, 4}, {4, 5, 6}};
        A = new Matrix(aEntries);
        assertEquals(false, A.isTri());
        assertEquals(false, A.isTriL());
        assertEquals(false, A.isTriU());
        assertEquals(false, A.isDiag());

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{
                {1, 3, 4},
                {4, 5, 6},
                {9, 13.334, -8.123}};
        A = new Matrix(aEntries);
        assertEquals(false, A.isTri());
        assertEquals(false, A.isTriL());
        assertEquals(false, A.isTriU());
        assertEquals(false, A.isDiag());

        // ----------------- Sub-case 3 -----------------
        aEntries = new double[][]{
                {1, 0, 0},
                {4, 5, 0},
                {9, 13.334, -8.123}};
        A = new Matrix(aEntries);
        assertEquals(true, A.isTri());
        assertEquals(true, A.isTriL());
        assertEquals(false, A.isTriU());
        assertEquals(false, A.isDiag());

        // ----------------- Sub-case 4 -----------------
        aEntries = new double[][]{
                {1, 3, 4},
                {0, 5, 6},
                {0, 0, -8.123}};
        A = new Matrix(aEntries);
        assertEquals(true, A.isTri());
        assertEquals(false, A.isTriL());
        assertEquals(true, A.isTriU());
        assertEquals(false, A.isDiag());

        // ----------------- Sub-case 5 -----------------
        aEntries = new double[][]{
                {1, 0, 0},
                {0, 5, 0},
                {0, 0, -8.123}};
        A = new Matrix(aEntries);
        assertEquals(true, A.isTri());
        assertEquals(true, A.isTriL());
        assertEquals(true, A.isTriU());
        assertEquals(true, A.isDiag());
    }
}
