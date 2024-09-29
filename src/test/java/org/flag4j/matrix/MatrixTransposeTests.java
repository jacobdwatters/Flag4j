package org.flag4j.matrix;

import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixTransposeTests {
    Matrix A, expT, AT;
    double[][] aEntries, expEntries;


    @Test
    void transposeTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{
                {1, 2, 3},
                {4, 5, 6}};
        expEntries = new double[][]{
                {1, 4},
                {2, 5},
                {3, 6}};
        A = new Matrix(aEntries);
        expT = new Matrix(expEntries);
        AT = A.T();
        assertArrayEquals(expT.entries, AT.entries);
        assertEquals(expT.numRows(), AT.numRows());
        assertEquals(expT.numCols(), AT.numCols());
    }
}
