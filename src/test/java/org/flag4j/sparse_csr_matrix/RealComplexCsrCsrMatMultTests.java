package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RealComplexCsrCsrMatMultTests {

    static CsrCMatrixOld B;
    static CMatrixOld bDense;
    static CNumber[][] bEntries;
    static CsrMatrixOld A;
    static MatrixOld aDense;
    static double[][] aEntries;
    static CMatrixOld exp;
    static CsrCMatrixOld expCsr;

    private static void build(boolean... args) {
        bDense = new CMatrixOld(bEntries);
        B = bDense.toCsr();
        aDense = new MatrixOld(aEntries);
        A = aDense.toCsr();
        if(args.length ==0 || args[0]) {
            exp = aDense.mult(bDense);
            expCsr = exp.toCsr();
        }
    }


    @Test
    void multTests() {
        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80.1, 2.5)},
                {new CNumber(0), new CNumber(1.41, -92.2), new CNumber(0), new CNumber(0, 15.5), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.25, 23.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(-999.1155, 2.25), new CNumber(-1, 1)}};
        aEntries = new double[][]{
                {1.45, 0, 0, 0},
                {0, 0.3265, 2.5, 0},
                {0, 0, 0, 0},
                {0, 0.36597, 0, 0},
                {0.18312, 0, 0, 0},
                {0.40715, 0, 0, 6.1}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(-77.3, -15122.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0, 803.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.345, 58.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1.45, -23), new CNumber(0)},
                {new CNumber(345), new CNumber(2.4, 5.61), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(4.45, -67.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        aEntries = new double[][]{
                {12.5, 0, 0, 0, 0, 53.67, 0, 0, 0, 2, 0},
                {0, -9.215, 0, 0, 851.3, 0, 0, 0, 0, 0, 5.15},
                {0, 0, 0, 0, 0, 0, 0, -481.3, 0, 0, 0},
                {1, 0, 0, 0, 6.3, 0, 0, 0, 0, 5, 0}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(-77.3, -15122.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0, 803.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.345, 58.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1.45, -23), new CNumber(0)},
                {new CNumber(345), new CNumber(2.4, 5.61), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(4.45, -67.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        aEntries = new double[][]{
                {0, 0.90836, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 8.3, 0},
                {0.23691, 0, 4.1, 0, 739.15, 0, 0, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()-> A.mult(B));
    }


    @Test
    void mult2CsrTests() {
        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80.1, 2.5)},
                {new CNumber(0), new CNumber(1.41, -92.2), new CNumber(0), new CNumber(0, 15.5), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.25, 23.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(-999.1155, 2.25), new CNumber(-1, 1)}};
        aEntries = new double[][]{
                {1.45, 0, 0, 0},
                {0, 0.3265, 2.5, 0},
                {0, 0, 0, 0},
                {0, 0.36597, 0, 0},
                {0.18312, 0, 0, 0},
                {0.40715, 0, 0, 6.1}};
        build();

        assertEquals(expCsr, A.mult2CSR(B));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(-77.3, -15122.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0, 803.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.345, 58.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1.45, -23), new CNumber(0)},
                {new CNumber(345), new CNumber(2.4, 5.61), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(4.45, -67.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        aEntries = new double[][]{
                {12.5, 0, 0, 0, 0, 53.67, 0, 0, 0, 2, 0},
                {0, -9.215, 0, 0, 851.3, 0, 0, 0, 0, 0, 5.15},
                {0, 0, 0, 0, 0, 0, 0, -481.3, 0, 0, 0},
                {1, 0, 0, 0, 6.3, 0, 0, 0, 0, 5, 0}};
        build();

        assertEquals(expCsr, A.mult2CSR(B));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(-77.3, -15122.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0, 803.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.345, 58.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1.45, -23), new CNumber(0)},
                {new CNumber(345), new CNumber(2.4, 5.61), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(4.45, -67.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        aEntries = new double[][]{
                {0, 0.90836, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 8.3, 0},
                {0.23691, 0, 4.1, 0, 739.15, 0, 0, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()-> A.mult2CSR(B));
    }
}
