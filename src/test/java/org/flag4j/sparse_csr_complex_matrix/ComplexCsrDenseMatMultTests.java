package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;

class ComplexCsrDenseMatMultTests {
    static CsrCMatrix A;
    static CMatrix aDense;
    static Complex128[][] aEntries;
    static Matrix B;
    static double[][] bEntries;
    static CMatrix exp;

    private static void build(boolean... args) {
        aDense = new CMatrix(aEntries);
        A = aDense.toCsr();
        B = new Matrix(bEntries);
        if(args.length != 1 || args[0]) exp = aDense.mult(B);
    }


    @Test
    void multTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(80.1, 2.5)},
                {new Complex128(0), new Complex128(1.41, -92.2), new Complex128(0), new Complex128(0, 15.5), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.25, 23.5), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(-999.1155, 2.25), new Complex128(-1, 1)}};
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
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541},
                {0.6462, 0.36597}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(-77.3, -15122.1), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0, 803.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(-9.345, 58.1), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(1.45, -23), new Complex128(0)},
                {new Complex128(345), new Complex128(2.4, 5.61), new Complex128(0), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(4.45, -67.2), new Complex128(0)},
                {new Complex128(0), new Complex128(0), new Complex128(0), new Complex128(1)}};
        bEntries = new double[][]{
                {0.72773, 0.90836},
                {0.02926, 0.3265},
                {0.23691, 0.77541}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
