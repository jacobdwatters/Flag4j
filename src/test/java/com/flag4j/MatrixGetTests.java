package com.flag4j;


import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixGetTests {

    static double[][] aEntries;
    double[][] expEntries;
    static Matrix A;
    Matrix exp;

    @BeforeAll
    static void setup() {
        aEntries = new double[][]{{1.123, 5525, 66.74}, {-8234.5, 15.22, -84.12}, {234, 8, 1}, {-9.451, -45.6, 111.345}};
        A = new Matrix(aEntries);
    }


    @Test
    void getRowTest() {
        // ------------------- Sub-case 1 -------------------
        exp = new Matrix(new double[][]{{-8234.5, 15.22, -84.12}});
        assertEquals(exp, A.getRow(1));

        // ------------------- Sub-case 2 -------------------
        exp = new Matrix(new double[][]{{1.123, 5525, 66.74}});
        assertEquals(exp, A.getRow(0));

        // ------------------- Sub-case 3 -------------------
        exp = new Matrix(new double[][]{{-9.451, -45.6, 111.345}});
        assertEquals(exp, A.getRow(3));

        // ------------------- Sub-case 4 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRow(13));

        // ------------------- Sub-case 5 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRow(-1));
    }


    @Test
    void getRowAsVectorTest() {
        Vector exp;

        // ------------------- Sub-case 1 -------------------
        exp = new Vector(new double[]{-8234.5, 15.22, -84.12});
        assertEquals(exp, A.getRowAsVector(1));

        // ------------------- Sub-case 2 -------------------
        exp = new Vector(new double[]{1.123, 5525, 66.74});
        assertEquals(exp, A.getRowAsVector(0));

        // ------------------- Sub-case 3 -------------------
        exp = new Vector(new double[]{-9.451, -45.6, 111.345});
        assertEquals(exp, A.getRowAsVector(3));

        // ------------------- Sub-case 4 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRowAsVector(13));

        // ------------------- Sub-case 5 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRowAsVector(-1));
    }


    @Test
    void getRowAfterTest() {
        // ------------------- Sub-case 1 -------------------
        exp = new Matrix(new double[][]{{15.22, -84.12}});
        assertEquals(exp, A.getRowAfter(1, 1));

        // ------------------- Sub-case 2 -------------------
        exp = new Matrix(new double[][]{{66.74}});
        assertEquals(exp, A.getRowAfter(2, 0));

        // ------------------- Sub-case 3 -------------------
        exp = new Matrix(new double[][]{{-9.451, -45.6, 111.345}});
        assertEquals(exp, A.getRowAfter(0, 3));

        // ------------------- Sub-case 4 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRowAfter(0, 13));

        // ------------------- Sub-case 5 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRowAfter(1, -1));

        // ------------------- Sub-case 6 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRowAfter(-1, 1));

        // ------------------- Sub-case 7 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getRowAfter(4, 1));
    }


    @Test
    void getSliceTest() {
        // ------------------- Sub-case 1 -------------------
        expEntries = new double[][]{{-8234.5, 15.22}, {234, 8}, {-9.451, -45.6}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.getSlice(1, 4, 0, 2));


        // ------------------- Sub-case 2 -------------------
        expEntries = new double[][]{{15.22}, {8}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.getSlice(1, 3, 1, 2));
    }


    @Test
    void getColAsVectorTest() {
        Vector exp;

        // ------------------- Sub-case 1 -------------------
        exp = new Vector(new double[]{5525, 15.22, 8, -45.6});
        assertEquals(exp, A.getColAsVector(1));

        // ------------------- Sub-case 2 -------------------
        exp = new Vector(new double[]{1.123, -8234.5, 234, -9.451});
        assertEquals(exp, A.getColAsVector(0));

        // ------------------- Sub-case 3 -------------------
        exp = new Vector(new double[]{66.74, -84.12, 1, 111.345});
        assertEquals(exp, A.getColAsVector(2));

        // ------------------- Sub-case 4 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getColAsVector(13));

        // ------------------- Sub-case 5 -------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.getColAsVector(-1));
    }
}