package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexCsrCsrMatMultTests {
    static CsrCMatrix A;
    static CMatrix aDense;
    static CNumber[][] aEntries;
    static CsrMatrix B;
    static Matrix bDense;
    static double[][] bEntries;
    static CMatrix exp;

    private static void build(boolean... args) {
        aDense = new CMatrix(aEntries);
        A = aDense.toCsr();
        bDense = new Matrix(bEntries);
        B = bDense.toCsr();
        if(args.length != 1 || args[0]) exp = aDense.mult(bDense);
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
                {1.45, 0},
                {0, 0.3265},
                {0, 0},
                {0, 0.36597},
                {0.18312, 0},
                {0.40715, 0}};
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
                {12.5, 0},
                {0, -9.215},
                {0, 0},
                {1, 0}};
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
                {0, 0.90836},
                {0, 0},
                {0.23691, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
