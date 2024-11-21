package org.flag4j.complex_matrix;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.junit.jupiter.api.Test;

class CMatrixZeroOnesTests {

    Complex128[] aEntries;
    CMatrix A;
    boolean exp;

    @Test
    void zerosTestCase()  {
        // ----------------- Sub-case 1 -----------------
        A = new CMatrix(14, 567);
        exp = true;

        assertEquals(exp, A.isZeros());

        // ----------------- Sub-case 2 -----------------
        A = new CMatrix(14, 567);
        A.set(new Complex128(-943, 133.5), 4, 5);
        exp = false;

        assertEquals(exp, A.isZeros());
    }


    @Test
    void onesTestCase()  {
        // ----------------- Sub-case 1 -----------------
        A = new CMatrix(14, 567, 1);
        exp = true;

        assertEquals(exp, A.isOnes());

        // ----------------- Sub-case 2 -----------------
        A = new CMatrix(14, 567, 1);
        A.set(new Complex128(-943, 133.5), 4, 55);
        exp = false;

        assertEquals(exp, A.isOnes());
    }
}
