package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixZeroOnesTests {

    Complex128[] aEntries;
    CMatrix A;
    boolean exp;

    @Test
    void zerosTestCase()  {
        // ----------------- sub-case 1 -----------------
        A = new CMatrix(14, 567);
        exp = true;

        assertEquals(exp, A.isZeros());

        // ----------------- sub-case 2 -----------------
        A = new CMatrix(14, 567);
        A.set(new Complex128(-943, 133.5), 4, 5);
        exp = false;

        assertEquals(exp, A.isZeros());
    }


    @Test
    void onesTestCase()  {
        // ----------------- sub-case 1 -----------------
        A = new CMatrix(14, 567, 1);
        exp = true;

        assertEquals(exp, A.isOnes());

        // ----------------- sub-case 2 -----------------
        A = new CMatrix(14, 567, 1);
        A.set(new Complex128(-943, 133.5), 4, 55);
        exp = false;

        assertEquals(exp, A.isOnes());
    }
}
