package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ComplexConversionTests {

    Matrix A;
    double[][] aEntries;
    CMatrix B, exp, act;
    CNumber[][] expEntries;


    @Test
    void toComplexTest() {
        aEntries = new double[][]
                {{1, 2, 0.912334, Double.NaN},
                {Double.POSITIVE_INFINITY, -9.322, -1992, 6434.2445}};
        A = new Matrix(aEntries);
        expEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2), new CNumber(0.912334), new CNumber(Double.NaN)},
                {new CNumber(Double.POSITIVE_INFINITY), new CNumber(-9.322), new CNumber(-1992), new CNumber(6434.2445)}};
        exp = new CMatrix(expEntries);
        act = A.toComplex();

        for(int i=0; i<exp.numRows(); i++) {
            for(int j=0; j<exp.numCols(); j++) {
                if(!Double.isNaN(exp.entries[i][j].re)) {
                    assertEquals(exp.entries[i][j], act.entries[i][j]);
                } else {
                    assertTrue(Double.isNaN(act.entries[i][j].re));
                    assertEquals(0, act.entries[i][j].im);
                }
            }
        }
    }

    static class MatrixExponentTests {
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
}
