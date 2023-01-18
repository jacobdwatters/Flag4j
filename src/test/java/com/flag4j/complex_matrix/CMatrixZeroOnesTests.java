package com.flag4j.complex_matrix;

import com.flag4j.CMatrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixZeroOnesTests {

    CNumber[] aEntries;
    CMatrix A;
    boolean exp;

    @Test
    void zerosTest()  {
        // ----------------- Sub-case 1 -----------------
        A = new CMatrix(14, 567);
        exp = true;

        assertEquals(exp, A.isZeros());

        // ----------------- Sub-case 2 -----------------
        A = new CMatrix(14, 567);
        A.set(new CNumber(-943, 133.5), 4, 5);
        exp = false;

        assertEquals(exp, A.isZeros());
    }


    @Test
    void onesTest()  {
        // ----------------- Sub-case 1 -----------------
        A = new CMatrix(14, 567, 1);
        exp = true;

        assertEquals(exp, A.isOnes());

        // ----------------- Sub-case 2 -----------------
        A = new CMatrix(14, 567, 1);
        A.set(new CNumber(-943, 133.5), 4, 55);
        exp = false;

        assertEquals(exp, A.isOnes());
    }
}
