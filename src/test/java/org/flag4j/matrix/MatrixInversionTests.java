package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.linalg.Invert;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixInversionTests {

    double[][] aEntries, expEntries;
    MatrixOld A, exp;


    @Test
    void invTestCase() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{
                {2, 55, 8},
                {3, 1, 1},
                {5, 4, 4}
        };
        A = new MatrixOld(aEntries);
        expEntries = new double[][]{
                {0.0, 0.5714285714285714, -0.14285714285714285},
                {0.02127659574468085, 0.09726443768996962, -0.06686930091185411},
                {-0.02127659574468085, -0.8115501519756839, 0.49544072948328266}
        };
        exp = new MatrixOld(expEntries);

        assertEquals(exp, Invert.inv(A));


        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{
                {1, 2},
                {-2, -4}
        }; // This matrix is singular
        A = new MatrixOld(aEntries);

        assertThrows(RuntimeException.class, ()->Invert.inv(A));


        // --------------------- Sub-case 3 ---------------------
        aEntries = new double[][]{
                {2, 55, 8},
                {3, 1, 1}
        };
        A = new MatrixOld(aEntries);

        assertThrows(LinearAlgebraException.class, ()->Invert.inv(A));
        assertThrows(LinearAlgebraException.class, ()->Invert.inv(A.T()));
    }
}
