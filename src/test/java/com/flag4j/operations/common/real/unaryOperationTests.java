package com.flag4j.operations.common.real;


import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class unaryOperationTests {

    double[][] aEntries, expEntries;
    Matrix A, exp;

    @Test
    void sqrtTest() {
        // ---------------- Sub-case 1 ----------------
        aEntries = new double[][]{{1.334, 5, 16}, {-134.5, 66.236, 144}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{Math.sqrt(1.334), Math.sqrt(5), Math.sqrt(16)},
                {Math.sqrt(-134.5), Math.sqrt(66.236), Math.sqrt(144)}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.sqrt());
    }

    @Test
    void absTest() {
        // ---------------- Sub-case 1 ----------------
        aEntries = new double[][]{{-1.334, 5, 16}, {-134.5, 66.236, 144}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{1.334, 5, 16}, {134.5, 66.236, 144}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.abs());
    }
}
