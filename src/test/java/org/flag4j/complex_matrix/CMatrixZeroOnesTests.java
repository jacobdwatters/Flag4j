package org.flag4j.complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixZeroOnesTests {

    CNumber[] aEntries;
    CMatrixOld A;
    boolean exp;

    @Test
    void zerosTestCase()  {
        // ----------------- Sub-case 1 -----------------
        A = new CMatrixOld(14, 567);
        exp = true;

        assertEquals(exp, A.isZeros());

        // ----------------- Sub-case 2 -----------------
        A = new CMatrixOld(14, 567);
        A.set(new CNumber(-943, 133.5), 4, 5);
        exp = false;

        assertEquals(exp, A.isZeros());
    }


    @Test
    void onesTestCase()  {
        // ----------------- Sub-case 1 -----------------
        A = new CMatrixOld(14, 567, 1);
        exp = true;

        assertEquals(exp, A.isOnes());

        // ----------------- Sub-case 2 -----------------
        A = new CMatrixOld(14, 567, 1);
        A.set(new CNumber(-943, 133.5), 4, 55);
        exp = false;

        assertEquals(exp, A.isOnes());
    }
}
