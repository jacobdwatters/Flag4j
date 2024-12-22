package org.flag4j.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class MatrixAggregationTests {

    double[][] aEntries;
    Matrix A;
    Double expAg;
    int[] expIndices;

    @Test
    void sumTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 1.334+-2.3112+334.3+4.13+-35.33+6;
        Assertions.assertEquals(expAg, A.sum());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.sum());
    }


    @Test
    void minTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = -35.33;
        Assertions.assertEquals(expAg, A.min());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.min());
    }


    @Test
    void maxTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 334.3;
        Assertions.assertEquals(expAg, A.max());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.max());
    }


    @Test
    void maxAbsTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 334.3;
        Assertions.assertEquals(expAg, A.maxAbs());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.maxAbs());
    }


    @Test
    void minAbsTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expAg = 1.334;
        Assertions.assertEquals(expAg, A.minAbs());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expAg = 0.0;
        Assertions.assertEquals(expAg, A.minAbs());
    }


    @Test
    void argminTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expIndices = new int[]{1, 1};
        Assertions.assertArrayEquals(expIndices, A.argmin());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expIndices = new int[]{};
        Assertions.assertArrayEquals(expIndices, A.argmin());
    }


    @Test
    void argmaxTestCase() {
        // ----------- Sub-case 1 -----------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        expIndices = new int[]{0, 2};
        Assertions.assertArrayEquals(expIndices, A.argmax());

        // ----------- Sub-case 2 -----------
        aEntries = new double[][]{{}};
        A = new Matrix(aEntries);
        expIndices = new int[]{};
        Assertions.assertArrayEquals(expIndices, A.argmax());
    }
}
