package com.flag4j;


import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixGetTests {

    double[][] aEntries, expEntries;
    Matrix A, exp;

    @Test
    void getSliceTest() {
        // ------------------- Sub-case 1 -------------------
        aEntries = new double[][]{{1.123, 5525, 66.74}, {-8234.5, 15.22, -84.12}, {234, 8, 1}, {-9.451, -45.6, 111.345}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{-8234.5, 15.22}, {234, 8}, {-9.451, -45.6}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.getSlice(1, 4, 0, 2));


        // ------------------- Sub-case 2 -------------------
        aEntries = new double[][]{{1.123, 5525, 66.74}, {-8234.5, 15.22, -84.12}, {234, 8, 1}, {-9.451, -45.6, 111.345}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{15.22}, {8}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.getSlice(1, 3, 1, 2));
    }
}
