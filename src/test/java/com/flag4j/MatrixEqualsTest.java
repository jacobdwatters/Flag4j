package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixEqualsTest {

    Matrix A, B;
    CMatrix BComplex;
    CNumber[][] bComplexEntries;
    double[][] aEntries, bEntries;
    boolean exp, act;


    @Test
    void eqTest() {
        // ----------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = true;
        act = A.equals(B);
        assertEquals(exp, act);

        // ----------------- Sub-case 2 --------------------
        aEntries = new double[][]{{1, 2, 3, 0.345}, {4, 5, 6, 9.2}};
        bEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = false;
        act = A.equals(B);
        assertEquals(exp, act);

        // ----------------- Sub-case 3 --------------------
        aEntries = new double[][]{{-1.234}, {4.556}};
        bEntries = new double[][]{{-1.234, 4.556}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = false;
        act = A.equals(B);
        assertEquals(exp, act);

        // ----------------- Sub-case 4 --------------------
        aEntries = new double[][]{{-0.2343, 1113.445, 9.234}, {0.89843, 899.442, 9934.1}};
        bEntries = new double[][]{{-0.2343, 1113.445, 9.234}, {0.89843, 899.442, 9934.1}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = true;
        act = A.equals(B);
        assertEquals(exp, act);

        // ----------------- Sub-case 5 --------------------
        aEntries = new double[][]{{-0.2343, 1113.445, 9.234},
                {0.89843, 899.442, 9934.1},
                {93.1, Double.POSITIVE_INFINITY, 1},
                {80.13, 3, Double.NEGATIVE_INFINITY}};
        bEntries = new double[][]{{-0.2343, 1113.445, 9.234},
                {0.89843, 899.442, 9934.1},
                {93.1, Double.POSITIVE_INFINITY, 1},
                {80.13, 3, Double.NEGATIVE_INFINITY}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = true;
        act = A.equals(B);
        assertEquals(exp, act);

        // ----------------- Sub-case 6 --------------------
        aEntries = new double[][]{{-0.2343, 1113.445, 9.234},
                {0.89843, 899.442, 9934.1},
                {93.1, Double.NaN, 1},
                {80.13, 3, 6}};
        bEntries = new double[][]{{-0.2343, 1113.445, 9.234},
                {0.89843, 899.442, 9934.1},
                {93.1, Double.NaN, 1},
                {80.13, 3, 6}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = false;
        act = A.equals(B);
        assertEquals(exp, act);
    }


    @Test
    void eqComplexTest() {
        // ----------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, -5, 6}, {7, 8, 9}, {10, 11, 12}};
        bComplexEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(4), new CNumber(-5), new CNumber(6)},};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = false;
        act = A.equals(BComplex);
        assertEquals(exp, act);

        // ----------------- Sub-case 2 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, -5, 6}, {7, 8, 9}};
        bComplexEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2), new CNumber(3)},
                        {new CNumber(4), new CNumber(-5), new CNumber(6)},
                        {new CNumber(7), new CNumber(8), new CNumber(9)}};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = true;
        act = A.equals(BComplex);
        assertEquals(exp, act);

        // ----------------- Sub-case 3 --------------------
        aEntries = new double[][]{{-1.234}, {4.556}};
        bComplexEntries = new CNumber[][]{{new CNumber(-1.234), new CNumber(4.556)}};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = false;
        act = A.equals(BComplex);
        assertEquals(exp, act);

        // ----------------- Sub-case 4 --------------------
        aEntries = new double[][]{{-0.2343, 1113.445, 9.234}, {0.89843, 899.442, 9934.1}};
        bComplexEntries = new CNumber[][]{{new CNumber(-0.2343), new CNumber(1113.445), new CNumber(9.234)},
                {new CNumber(0.89843), new CNumber(899.442), new CNumber(9934.1)}};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = true;
        act = A.equals(BComplex);
        assertEquals(exp, act);

        // ----------------- Sub-case 5 --------------------
        aEntries = new double[][]{{-0.2343, 1113.445, 9.234},
                {0.89843, 899.442, 9934.1},
                {93.1, Double.POSITIVE_INFINITY, 1},
                {80.13, 3, Double.NEGATIVE_INFINITY}};
        bComplexEntries = new CNumber[][]
                {{new CNumber(-0.2343), new CNumber(1113.445), new CNumber(9.234)},
                {new CNumber(0.89843), new CNumber(899.442), new CNumber(9934.1)},
                {new CNumber(93.1), new CNumber(Double.POSITIVE_INFINITY), new CNumber(1)},
                {new CNumber(80.13), new CNumber(3), new CNumber(Double.NEGATIVE_INFINITY)}};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = true;
        act = A.equals(BComplex);
        assertEquals(exp, act);

        // ----------------- Sub-case 6 --------------------
        aEntries = new double[][]{{-0.2343, 1113.445, 9.234},
                {0.89843, 899.442, 9934.1},
                {93.1, Double.NaN, 1},
                {80.13, 3, 6}};
        bComplexEntries = new CNumber[][]
                {{new CNumber(-0.2343), new CNumber(1113.445), new CNumber(9.234)},
                {new CNumber(0.89843), new CNumber(899.442), new CNumber(9934.1)},
                {new CNumber(93.1), new CNumber(Double.NaN), new CNumber(1)},
                {new CNumber(80.13), new CNumber(3), new CNumber(6)}};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = false;
        act = A.equals(BComplex);
        assertEquals(exp, act);

        // ----------------- Sub-case 7 --------------------
        aEntries = new double[][]{{1, 2}, {3, 4}};
        bComplexEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2, 1)},
                {new CNumber(3), new CNumber(4)}};
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);
        exp = false;
        act = A.equals(BComplex);
        assertEquals(exp, act);
    }
}
