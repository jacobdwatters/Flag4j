package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrMatrixTransposeTests {

    static CsrMatrix A;
    static CsrMatrix exp;
    static Matrix denseA;
    static double[][] aDenseEntries;


    private static void makeMatrices() {
        denseA = new Matrix(aDenseEntries);
        exp = denseA.T().toCsr();
        A = denseA.toCsr();
    }


    @Test
    void transposeTest() {
        // --------------------- Sub-case 1 ----------------------
        aDenseEntries = new double[][]{
                {0.245, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 9, 0, 0, 0},
                {0, 0, 1.14, 0, 0, 0, 0, 24.1},
                {0, 0, 0, 12, 0, 0, -93.51134, 0},
                {0, 0, 0, 15.14, 24.15, 0, 0, -0.114}
        };
        makeMatrices();
        assertEquals(exp, A.T());

        // --------------------- Sub-case 2 ----------------------
        aDenseEntries = new double[][]{
                {0, 0, 1, 0},
                {0, -25, 0, 0},
                {0, 4, 0, 0},
                {0, 12, 0, 0},
                {0, 14.15, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {-1, 92, 15.2, 7},
                {0, 0, 0, 8.13}
        };
        makeMatrices();
        assertEquals(exp, A.T());
    }
}
