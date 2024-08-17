package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixScalMultTests {
    double[][] aEntries, expEntries;
    CNumber[][] expCEntries;
    MatrixOld A, exp;
    CMatrixOld expC;
    CNumber aC;
    double a;


    @Test
    void realTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new MatrixOld(aEntries);
        a = -1.66623;
        expEntries = new double[][]{{1*-1.66623, 2*-1.66623, 3.3434*-1.66623},
                {-0.221*-1.66623, 81.346*-1.66623, 90.234*-1.66623}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, A.mult(a));

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new MatrixOld(aEntries);
        a = 0;
        expEntries = new double[2][3];
        expEntries[1][0] = -0.0;
        exp = new MatrixOld(expEntries);

        assertEquals(exp, A.mult(a));
    }


    @Test
    void complexTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new MatrixOld(aEntries);
        aC = new CNumber(-9.123, 19.23);
        expCEntries = new CNumber[][]{
                {new CNumber(1).mult(aC), new CNumber(2).mult(aC), new CNumber(3.3434).mult(aC)},
                {new CNumber(-0.221).mult(aC), new CNumber(81.346).mult(aC), new CNumber(90.234).mult(aC)}};
        expC = new CMatrixOld(expCEntries);

        assertEquals(expC, A.mult(aC));

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3.3434}, {-0.221, 81.346, 90.234}};
        A = new MatrixOld(aEntries);
        aC = new CNumber(0);
        expCEntries = new CNumber[][]{
                {new CNumber(1).mult(aC), new CNumber(2).mult(aC), new CNumber(3.3434).mult(aC)},
                {new CNumber(-0.221).mult(aC), new CNumber(81.346).mult(aC), new CNumber(90.234).mult(aC)}};
        expC = new CMatrixOld(expCEntries);

        assertEquals(expC, A.mult(aC));
    }
}
