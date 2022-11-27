package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixSymmetryTests {

    Matrix A;
    double[][] aEntries;
    boolean exp, act;

    @Test
    void symmetricTest() {
        // ----------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = false;
        act = A.isSymmetric();
        assertEquals(exp, act);

        // ----------- Sub-case 2 --------------
        aEntries = new double[][]
                {{1, 2, 3.33},
                 {2, 59234.133, -6},
                 {3.33, -6, 5.221}};
        A = new Matrix(aEntries);
        exp = true;
        act = A.isSymmetric();
        assertEquals(exp, act);

        // ----------- Sub-case 3 --------------
        aEntries = new double[][]
                {{1, 2, 3.33},
                        {2, 59234.133, -6}};
        A = new Matrix(aEntries);
        exp = false;
        act = A.isSymmetric();
        assertEquals(exp, act);

        // ----------- Sub-case 4 --------------
        aEntries = new double[100][101];
        A = new Matrix(aEntries);
        exp = false;
        act = A.isSymmetric();
        assertEquals(exp, act);
    }


    @Test
    void antiSymmetricTest() {
        // ----------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp = false;
        act = A.isAntiSymmetric();
        assertEquals(exp, act);

        // ----------- Sub-case 2 --------------
        aEntries = new double[][]
                {{1, 2, 3.33},
                {-2, -59234.133, -6},
                {-3.33, 6, 5.221}};
        A = new Matrix(aEntries);
        exp = true;
        act = A.isAntiSymmetric();
        assertEquals(exp, act);

        // ----------- Sub-case 3 --------------
        aEntries = new double[][]
                {{1, 2, 3.33},
                {-2, -59234.133, -6}};
        A = new Matrix(aEntries);
        exp = false;
        act = A.isAntiSymmetric();
        assertEquals(exp, act);

        // ----------- Sub-case 4 --------------
        aEntries = new double[100][101];
        A = new Matrix(aEntries);
        exp = false;
        act = A.isAntiSymmetric();
        assertEquals(exp, act);
    }
}
