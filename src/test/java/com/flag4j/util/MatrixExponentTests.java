package com.flag4j.util;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixExponentTests {
    Matrix A;
    Matrix exp, act;
    CMatrix expComplex, actComplex;
    double[][] aEntries, expEntries;
    CNumber[][] expComplexEntries;
    double power;

    @Test
    void sqrtTest() {
        aEntries = new double[][]{{1, 25, 3}, {4, -5, 61.234}};
        expEntries = new double[][]{{1, 5, Math.sqrt(3)}, {2, Double.NaN, Math.sqrt(61.234)}};

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.sqrt();

        assertArrayEquals(exp.entries, act.entries);
    }


    @Test
    void sqrtComplexTest() {
        aEntries = new double[][]{{1, -1, 25}, {441.3, -5, -61.234}};
        expComplexEntries = new CNumber[][]{{new CNumber("1"), new CNumber("i"), new CNumber(Math.sqrt(25))},
                {new CNumber(Math.sqrt(441.3)), new CNumber(0, Math.sqrt(5)), new CNumber(0, Math.sqrt(61.234))}};

        A = new Matrix(aEntries);
        expComplex = new CMatrix(expComplexEntries);

        actComplex = A.sqrtComplex();

        assertArrayEquals(expComplex.entries, actComplex.entries);
    }


    @Test
    void powTest() {
        // -------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 5, 3}, {4, -5, 61.234}};
        expEntries = new double[][]{{1, 25, 9}, {16, 25, Math.pow(61.234, 2)}};
        power = 2;

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.pow(power);

        assertArrayEquals(exp.entries, act.entries);


        // -------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 5, 3}, {4, -5, 61.234}};
        expEntries = new double[][]{{Math.pow(1, -4.234), Math.pow(5, -4.234),Math.pow(3, -4.234)},
                {Math.pow(4, -4.234), Math.pow(-5, -4.234), Math.pow(61.234, -4.234)}};
        power = -4.234;

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.pow(power);

        assertArrayEquals(exp.entries, act.entries);
    }
}
