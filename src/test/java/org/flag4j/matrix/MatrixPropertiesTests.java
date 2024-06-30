package org.flag4j.matrix;

import org.flag4j.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixPropertiesTests {

    double[][] aEntries;
    Matrix A;
    boolean expBoolResult;

    @Test
    void isIdentityTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {-0.442, 13.5, 35.6}, {0.4441, 6, 90}};
        A = new Matrix(aEntries);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {-0.442, 13.5, 35.6}};
        A = new Matrix(aEntries);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 3 ---------------
        aEntries = new double[][]{{1, 2}, {-0.442, 13.5}};
        A = new Matrix(aEntries);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 4 ---------------
        aEntries = new double[][]{{1, 0},
                                  {0, 1}};
        A = new Matrix(aEntries);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 5 ---------------
        aEntries = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        A = new Matrix(aEntries);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 6 ---------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        A = new Matrix(aEntries);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 6 ---------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        A = new Matrix(aEntries);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 7 ---------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 1}};
        A = new Matrix(aEntries);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 8 ---------------
        aEntries = new double[][]{{0, 0}, {0, 0}};
        A = new Matrix(aEntries);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());
    }


    @Test
    void isInvTestCase() {
        double[][] bEntries;
        Matrix B;

        // ------------------ Sub-case 1 ------------------
        aEntries = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        B = new Matrix(bEntries);

        assertTrue(A.isInv(B));

        // ------------------ Sub-case 2 ------------------
        aEntries = new double[][]{{2, 1}, {7, 4}};
        A = new Matrix(aEntries);

        bEntries = new double[][]{{4, -1}, {-7, 2}};
        B = new Matrix(bEntries);

        assertTrue(A.isInv(B));


        // ------------------ Sub-case 3 ------------------
        aEntries = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        B = new Matrix(bEntries);

        assertTrue(A.isInv(B));

        // ------------------ Sub-case 4 ------------------
        aEntries = new double[][]{{2.243, 1}, {7235, 4.44656}};
        A = new Matrix(aEntries);

        bEntries = new double[][]{{4, -1.234432}, {-7, 2}};
        B = new Matrix(bEntries);

        assertFalse(A.isInv(B));

        // ------------------ Sub-case 5 ------------------
        aEntries = new double[][]{{2.243, 1,23.4}, {7235, 4.44656, -845.5}};
        A = new Matrix(aEntries);

        bEntries = new double[][]{{4, -1.234432, -6}, {-7, 2, 234.5}};
        B = new Matrix(bEntries);

        assertFalse(A.isInv(B));
    }


    @Test
    void isOrthogonalTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {-0.442, 13.5, 35.6}, {0.4441, 6, 90}};
        A = new Matrix(aEntries);
        assertFalse(A.isOrthogonal());

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}};
        A = new Matrix(aEntries);
        assertTrue(A.isOrthogonal());

        // --------------- Sub-case 3 ---------------
        aEntries = new double[][]{
                {1, 0, 0},
                {0, 0, 1},
                {0, 1, 0}};
        A = new Matrix(aEntries);
        assertTrue(A.isOrthogonal());

        // --------------- Sub-case 4 ---------------
        aEntries = new double[][]{
                {2, -2, 1},
                {1, 2, 2},
                {2, 1, -2}};
        A = new Matrix(aEntries).div(3);
        assertTrue(A.isOrthogonal());

        // --------------- Sub-case 4 ---------------
        aEntries = new double[][]{
                {-0.6372525669012274, 0.44183946030385846, 0.5401926054008803, -0.031537469407094426, -0.3253988235839168},
                {-0.4244602495444651, -0.278448048912439, -0.384432919721785, 0.7274699431594314, -0.2555366757381484},
                {-0.605719911072018, 0.02593952365425975, -0.3924026386562259, -0.3390849382170871, 0.6028864771194014},
                {-0.16792485508361002, -0.5127194029812936, -0.16381749990785172, -0.5854281629671453, -0.5825442039712599},
                {-0.13655509844199198, -0.6809482581623562, 0.6161065453344614, 0.10994607997667842, 0.35494613547543796}};
        assertTrue(A.isOrthogonal());

        // --------------- Sub-case 5 ---------------
        aEntries = new double[][]{
                {2, -2, 1},
                {1, 2, 2}};
        A = new Matrix(aEntries).div(3);
        assertFalse(A.isOrthogonal());
    }
}
