package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrMatrixToDenseTests {
    static CsrMatrixOld A;
    static double[][] aEntries;

    static MatrixOld exp;

    static void build() {
        A = new MatrixOld(aEntries).toCsr();
        exp = new MatrixOld(aEntries);
    }


    @Test
    void toDenseTests() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.67525},
                {0.77089, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.10858, 0.17215},
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.47055, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.7428},
                {0.0, 0.0, 0.0, 0.0}};
        build();

        assertEquals(exp, A.toDense());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.35447, 0.0, 0.44042, 0.0, 0.86769, 0.38842},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.77065},
                {0.0, 0.0, 0.0, 0.0, 0.39301, 0.0, 0.0}};
        build();

        assertEquals(exp, A.toDense());
    }
}
