package com.flag4j;

import com.flag4j.Matrix;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixZerosOnesTests {

    double[][] aEntries;
    Matrix A;
    boolean exp;

    @Test
    void zerosTest()  {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[46][101];
        A = new Matrix(aEntries);
        exp = true;

        assertEquals(exp, A.isZeros());

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[46][101];
        aEntries[21][9] = 1.324;
        A = new Matrix(aEntries);
        exp = false;

        assertEquals(exp, A.isZeros());
    }


    @Test
    void onesTest()  {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[46][101];
        ArrayUtils.fill(aEntries, 1.0);
        A = new Matrix(aEntries);
        exp = true;

        assertEquals(exp, A.isOnes());

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[46][101];
        ArrayUtils.fill(aEntries, 1.0);
        aEntries[21][9] = -1;
        A = new Matrix(aEntries);
        exp = false;

        assertEquals(exp, A.isOnes());
    }
}
