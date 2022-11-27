package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ElementwisePropertyTests {

    Matrix A;
    double[][] aEntries;
    boolean exp, act;

    @Test
    void isZeroTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{0, 0, 0, 1}, {0, 0, 0, 0}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, 234}, {0, 0}, {0, 13}, {-9.3, 0813.3}, {Double.NaN}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{0, 0, 0, Double.NaN}, {0, 0, 0, 0}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{0, 0, 0, 0}, {0, 0, 0, 0}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        aEntries = new double[][]{{0}, {0}, {0}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

        // -------------- Sub-case 6 --------------
        aEntries = new double[832][346];
        exp = true;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

        // -------------- Sub-case 7 --------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isZeros();
        assertEquals(exp, act);

    }


    @Test
    void isOneTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 1, 1, 1}, {1, 0, 1, 1}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isOnes();
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, 234}, {0, 0}, {0, 13}, {-9.3, 0813.3}, {Double.NaN}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isOnes();
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{1, 1, 1, Double.NaN}, {1, 1, 1, 1}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isOnes();
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{1, 1, 1, 1}, {1, 1, 1, 1}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isOnes();
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        aEntries = new double[][]{{1}, {1}, {1}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isOnes();
        assertEquals(exp, act);

        // -------------- Sub-case 6 --------------
        aEntries = new double[832][346];
        exp = false;
        A = new Matrix(aEntries);
        act = A.isOnes();
        assertEquals(exp, act);
    }


    @Test
    void isPosTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{0, 0, 0, 1}, {0, 1334, 0, 0}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, 234}, {0, 0}, {0, 13}, {-9.3, 0813.3}, {Double.NaN}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{0, 0, 0, Double.NaN}, {0, 0, 0, 0}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{0, 0, 0, 0}, {0, 0, 0, 0}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        aEntries = new double[][]{{1, -2, 3}, {4, 5, 6}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 6 --------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 7 --------------
        aEntries = new double[][]{{1, Double.POSITIVE_INFINITY, 3}, {4, 5, 6}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);

        // -------------- Sub-case 8 --------------
        aEntries = new double[][]{{1, Double.NEGATIVE_INFINITY, 3}, {4, 5, 6}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isPos();
        assertEquals(exp, act);
    }


    @Test
    void isNegTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{0, 0, 0, -1}, {0, -1334, 0, 0}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, -234}, {0, 0}, {0, -13}, {-9.3, -0813.3}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{0, 0, 0, Double.NaN}, {0, 0, 0, 0}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{0, 0, 0, 0}, {0, 0, 0, 0}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        aEntries = new double[][]{{-1, -2, -3}, {4, -5, -6}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 6 --------------
        aEntries = new double[][]{{-1, -2, -3}, {-4, -5, -6}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 7 --------------
        aEntries = new double[][]{{-1, Double.POSITIVE_INFINITY, -3}, {-4, -5, -6}};
        exp = false;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);

        // -------------- Sub-case 8 --------------
        aEntries = new double[][]{{-1, Double.NEGATIVE_INFINITY, -3}, {-4, -5, -6}};
        exp = true;
        A = new Matrix(aEntries);
        act = A.isNeg();
        assertEquals(exp, act);
    }
}
