package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class MatrixZerosOnesTests {

    double[][] aEntries;
    MatrixOld A;
    boolean exp;

    @Test
    void zerosTestCase()  {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[46][101];
        A = new MatrixOld(aEntries);
        exp = true;

        Assertions.assertEquals(exp, A.isZeros());

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[46][101];
        aEntries[21][9] = 1.324;
        A = new MatrixOld(aEntries);
        exp = false;

        Assertions.assertEquals(exp, A.isZeros());
    }


    @Test
    void onesTestCase()  {
        // ----------------- Sub-case 1 -----------------
        aEntries = new double[46][101];
        ArrayUtils.fill(aEntries, 1.0);
        A = new MatrixOld(aEntries);
        exp = true;

        Assertions.assertEquals(exp, A.isOnes());

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[46][101];
        ArrayUtils.fill(aEntries, 1.0);
        aEntries[21][9] = -1;
        A = new MatrixOld(aEntries);
        exp = false;

        Assertions.assertEquals(exp, A.isOnes());
    }
}
