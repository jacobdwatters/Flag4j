package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixVectorTypeTests {
    Matrix A;
    double[][] aEntries;
    boolean expBool, actBool;
    int expInt, actInt;


    @Test
    void isVectorTest() {
        // -------------- Sub-case 1 -----------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        expBool = false;

        actBool = A.isVector();
        assertEquals(expBool, actBool);

        // -------------- Sub-case 2 -----------------
        aEntries = new double[][]{{1, 2, 3, 3, 45, 22.24343}};
        A = new Matrix(aEntries);
        expBool = true;

        actBool = A.isVector();
        assertEquals(expBool, actBool);

        // -------------- Sub-case 3 -----------------
        aEntries = new double[][]{{3}, {4}, {5}, {1}};
        A = new Matrix(aEntries);
        expBool = true;

        actBool = A.isVector();
        assertEquals(expBool, actBool);

        // -------------- Sub-case 4 -----------------
        aEntries = new double[][]{{3.14159}};
        A = new Matrix(aEntries);
        expBool = true;

        actBool = A.isVector();
        assertEquals(expBool, actBool);

        // -------------- Sub-case 5 -----------------
        aEntries = new double[234][1];
        A = new Matrix(aEntries);
        expBool = true;

        actBool = A.isVector();
        assertEquals(expBool, actBool);

        // -------------- Sub-case 6 -----------------
        aEntries = new double[2][3421];
        A = new Matrix(aEntries);
        expBool = false;

        actBool = A.isVector();
        assertEquals(expBool, actBool);
    }


    @Test
    void vectorTypeTest() {
        // -------------- Sub-case 1 -----------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        expInt = -1;
        A = new Matrix(aEntries);
        actInt = A.vectorType();
        assertEquals(expInt, actInt);

        // -------------- Sub-case 2 -----------------
        aEntries = new double[][]{{1, 2, 3.322343, -9079234.2}};
        expInt = 1;
        A = new Matrix(aEntries);
        actInt = A.vectorType();
        assertEquals(expInt, actInt);


        // -------------- Sub-case 3 -----------------
        aEntries = new double[][]{{1}};
        expInt = 0;
        A = new Matrix(aEntries);
        actInt = A.vectorType();
        assertEquals(expInt, actInt);


        // -------------- Sub-case 3 -----------------
        aEntries = new double[][]{{1}, {4}, {159}};
        expInt = 2;
        A = new Matrix(aEntries);
        actInt = A.vectorType();
        assertEquals(expInt, actInt);
    }
}
