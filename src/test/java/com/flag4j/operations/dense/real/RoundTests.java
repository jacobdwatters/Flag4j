package com.flag4j.operations.dense.real;

import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RoundTests {

    double[][] aEntries, expEntries;
    Matrix A, exp;
    int precision;

    @Test
    void simpleRoundTests() {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.556}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{1, 1257, -9991}, {115, 0, -6612355}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.round());
    }


    @Test
    void roundPrecisionTests() {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.556}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{1, 1257, -9991}, {115, 0, -6612355}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.round(0));

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.556}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{1, 1257, -9991}, {115, 0, -6612355}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.round(0));

        // ----------------- Sub-case 3 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.556}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{1.23, 1256.55, -9991.13}, {115, 0, -6612354.56}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.round(2));

        // ----------------- Sub-case 4 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.99989}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{1.23, 1256.55, -9991.13}, {115, 0, -6612355}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.round(2));

        // ----------------- Sub-case 5 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.99989}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.99989}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.round(25));

        // ----------------- Sub-case 5 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.99989}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()->A.round(-1));
    }


    @Test
    void simpleRoundToZeroTests() {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{{-0.0008915, 1256.55, -9991.133}, {115, 0.000014, -6612354.556},
                {0.00000000000008765, 0.013, 133.45}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{-0.0008915, 1256.55, -9991.133}, {115, 0.000014, -6612354.556},
                {0, 0.013, 133.45}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.roundToZero());

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{{-0.0008915, 1256.55, -9991.133}, {115, 0.000014, -6612354.556},
                {0.00000000000008765, 0.013, 133.45}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{-0.0008915, 1256.55, -9991.133}, {115, 0, -6612354.556},
                {0, 0.013, 133.45}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.roundToZero(0.0005));

        // ----------------- Sub-case 3 -----------------
        aEntries = new double[][]{{-0.0008915, 1256.55, -9991.133}, {115, 0.000014, -6612354.556},
                {0.00000000000008765, 0.013, 133.45}};
        A = new Matrix(aEntries);

        expEntries = new double[][]{{0, 1256.55, -9991.133}, {0, 0, -6612354.556},
                {0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.roundToZero(135));

        // ----------------- Sub-case 4 -----------------
        aEntries = new double[][]{{1.234234, 1256.55, -9991.133}, {115, 0.000014, -6612354.99989}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()->A.roundToZero(-1));
    }
}
