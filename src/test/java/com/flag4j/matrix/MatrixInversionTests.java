package com.flag4j.matrix;

import com.flag4j.Matrix;
import com.flag4j.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixInversionTests {

    double[][] aEntries, expEntries;
    Matrix A, exp;


    @Test
    void invTestCase() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{
                {2, 55, 8},
                {3, 1, 1},
                {5, 4, 4}
        };
        A = new Matrix(aEntries);
        expEntries = new double[][]{
                {0.0, 0.5714285714285714, -0.14285714285714285},
                {0.02127659574468085, 0.09726443768996962, -0.06686930091185411},
                {-0.02127659574468085, -0.8115501519756839, 0.49544072948328266}
        };
        exp = new Matrix(expEntries);

        assertEquals(exp, A.inv());


        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{
                {1, 2},
                {-2, -4}
        }; // This matrix is singular
        A = new Matrix(aEntries);

        assertThrows(RuntimeException.class, ()->A.inv());


        // --------------------- Sub-case 3 ---------------------
        aEntries = new double[][]{
                {2, 55, 8},
                {3, 1, 1}
        };
        A = new Matrix(aEntries);

        assertThrows(LinearAlgebraException.class, ()->A.inv());
        assertThrows(LinearAlgebraException.class, ()->A.T().inv());
    }
}
