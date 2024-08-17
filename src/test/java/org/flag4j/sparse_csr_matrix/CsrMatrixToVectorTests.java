package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrMatrixToVectorTests {
    CsrMatrix A;
    double[][] aEntries;

    CooVector exp;
    double[] expEntries;

    @Test
    void toVectorTests() {
        // ------------------------- Sub-case 1 -------------------------
        aEntries = new double[][]{{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[]{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2};
        exp = new VectorOld(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- Sub-case 2 -------------------------
        aEntries = new double[][]{{-9.2512}, {0}, {0},
                {0}, {0}, {2.2516},
                {0}, {0}, {6.2}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[]{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2};
        exp = new VectorOld(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- Sub-case 3 -------------------------
        aEntries = new double[][]{
                {-9.2512, 0},
                {0, 0},
                {0, 2.2516},
                {0, 0},
                {6.2, 0}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[]{-9.2512, 0, 0,
                0, 0, 2.2516,
                0, 0, 6.2, 0};
        exp = new VectorOld(expEntries).toCoo();

        assertEquals(exp, A.toVector());
    }
}
