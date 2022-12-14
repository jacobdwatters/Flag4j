package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixAggregationTests {

    double[][] aEntries;
    Matrix A;
    Double expAg;
    int expAgInt;

    @Test
    void sumTest() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 1.334+-2.3112+334.3+4.13+-35.33+6;
        assertEquals(expAg, A.sum());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        assertEquals(expAg, A.sum());
    }


    @Test
    void minTest() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = -35.33;
        assertEquals(expAg, A.min());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        assertEquals(expAg, A.min());
    }


    @Test
    void maxTest() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 334.3;
        assertEquals(expAg, A.max());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        assertEquals(expAg, A.max());
    }


    @Test
    void maxAbsTest() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 334.3;
        assertEquals(expAg, A.maxAbs());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        assertEquals(expAg, A.maxAbs());
    }


    @Test
    void minAbsTest() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 1.334;
        assertEquals(expAg, A.minAbs());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        assertEquals(expAg, A.minAbs());
    }
}
