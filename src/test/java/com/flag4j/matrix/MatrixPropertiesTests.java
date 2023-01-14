package com.flag4j.matrix;

import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixPropertiesTests {

    double[][] entriesA;
    Matrix A;
    boolean expBoolResult;

    @Test
    void isIdentityTest() {
        // --------------- Sub-case 1 ---------------
        entriesA = new double[][]{{1, 2, 3}, {-0.442, 13.5, 35.6}, {0.4441, 6, 90}};
        A = new Matrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 2 ---------------
        entriesA = new double[][]{{1, 2, 3}, {-0.442, 13.5, 35.6}};
        A = new Matrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 3 ---------------
        entriesA = new double[][]{{1, 2}, {-0.442, 13.5}};
        A = new Matrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 4 ---------------
        entriesA = new double[][]{{1, 0},
                                  {0, 1}};
        A = new Matrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 5 ---------------
        entriesA = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        A = new Matrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 6 ---------------
        entriesA = new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        A = new Matrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 6 ---------------
        entriesA = new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        A = new Matrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 7 ---------------
        entriesA = new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 1}};
        A = new Matrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 8 ---------------
        entriesA = new double[][]{{0, 0}, {0, 0}};
        A = new Matrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());
    }
}
