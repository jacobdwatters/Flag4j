package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexCsrDenseMatMultTests {
    static CsrCMatrixOld A;
    static CMatrixOld aDense;
    static CNumber[][] aEntries;
    static MatrixOld B;
    static double[][] bEntries;
    static CMatrixOld exp;

    private static void build(boolean... args) {
        aDense = new CMatrixOld(aEntries);
        A = aDense.toCsr();
        B = new MatrixOld(bEntries);
        if(args.length != 1 || args[0]) exp = aDense.mult(B);
    }


    @Test
    void multTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80.1, 2.5)},
                {new CNumber(0), new CNumber(1.41, -92.2), new CNumber(0), new CNumber(0, 15.5), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-9.25, 23.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(-999.1155, 2.25), new CNumber(-1, 1)}};
        bEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541},
                {0.6462, 0.36597},
                {0.18312, 0.77178},
                {0.40715, 0.35642}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
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
        bEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541},
                {0.6462, 0.36597}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
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
        bEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
