package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixAddSubTests {

    static CsrMatrix A;
    static CsrMatrix B;
    static Matrix denseA;
    static Matrix denseB;
    static CsrMatrix expAdd;
    static CsrMatrix expAsubB;
    static CsrMatrix expBsubA;
    static double[][] aEntries;
    static double[][] bEntries;

    private static void makeMatrices() {
        denseA = new Matrix(aEntries);
        denseB = new Matrix(bEntries);
        A = denseA.toCsr();
        B = denseB.toCsr();
        expAdd = denseA.add(denseB).toCsr();
        expAsubB = denseA.sub(denseB).toCsr();
        expBsubA = denseB.sub(denseA).toCsr();
    }

    @Test
    void addSubTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {0, 0, 1.34, 0, 1.3, 0},
                {0, 0, 0, 56.1, 0, 0},
                {36.1, 13.2, 0, 0, 0, 8},
                {0, 0, 0, 0, 0, 0},
        };
        makeMatrices();
        assertEquals(expAdd, A.add(B));
        assertEquals(expAsubB, A.sub(B));
        assertEquals(expBsubA, B.sub(A));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[2], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[5], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
        assertThrows(LinearAlgebraException.class, ()->B.add(A));
        assertThrows(LinearAlgebraException.class, ()->B.sub(A));
    }
}
