package com.flag4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixPropertiesTests {

    double[][] aEntries;
    Matrix A;
    boolean expBoolResult;

    @Test
    void isIdentityTest() {
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
    void isInvTest() {
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
}
