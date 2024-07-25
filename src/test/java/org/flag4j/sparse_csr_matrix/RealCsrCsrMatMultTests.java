package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RealCsrCsrMatMultTests {

    static CsrMatrix A;
    static Matrix aDense;
    static double[][] aEntries;
    static CsrMatrix B;
    static Matrix bDense;
    static double[][] bEntries;
    static Matrix exp;

    private static void build(boolean... args) {
        aDense = new Matrix(aEntries);
        A = aDense.toCsr();
        bDense = new Matrix(bEntries);
        B = bDense.toCsr();
        if(args.length != 1 || args[0]) exp = aDense.mult(bDense);
    }


    @Test
    void multTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 80.1},
                {0, 1.41, 0, 15.5, 0, 0},
                {0, -9.25, 0, 0, 0, 0},
                {0, 0, 0, 0, -999.1155, 1}};
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
        aEntries = new double[][]{
                {0, 0, 0, 0},
                {-77.3, 0, 0, 0},
                {0, 0, 803.2, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, -9.345, 0, 0},
                {0, 0, 1.45, 0},
                {345, 2.4, 0, 0},
                {0, 0, 4.45, 0},
                {0, 0, 0, 1},};
        bEntries = new double[][]{
                {12.5, 0},
                {0, -9.215},
                {0, 0},
                {1, 0}};
        build();

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0},
                {-77.3, 0, 0, 0},
                {0, 0, 803.2, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, -9.345, 0, 0},
                {0, 0, 1.45, 0},
                {345, 2.4, 0, 0},
                {0, 0, 4.45, 0},
                {0, 0, 0, 1},};
        bEntries = new double[][]{
                {0, 0.90836},
                {0, 0},
                {0.23691, 0}};
        build(false);

        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
