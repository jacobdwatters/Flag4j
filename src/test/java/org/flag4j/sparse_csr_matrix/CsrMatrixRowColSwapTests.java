package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixRowColSwapTests {

    CsrMatrixOld A;
    CsrMatrixOld exp;
    double[][] aEntries;
    double[][] expEntries;


    @Test
    void rowSwapTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 0, 0, 0, 0, 80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 0, 0, 0, 80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapRows(1, 2));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 0, 0, 0, 0, 80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 0, 0, 0, 80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapRows(2, 1));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 0, 0, 0, 0, 80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {0, 0, 50, 60, 70, 0},
                {0, 30, 0, 40, 0, 0},
                {10, 20, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapRows(2, 0));

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 0, 0, 0, 0, 80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {0, 0, 0, 0, 0, 80},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {10, 20, 0, 0, 0, 0}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapRows(0, 3));

        // ---------------------- Sub-case 5 ----------------------
        aEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 0, 0, 0, 0, 80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 0, 0, 0, 80},
                {0, 0, 50, 60, 70, 0}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapRows(3, 2));

        // ---------------------- Sub-case 6 ----------------------
        aEntries = new double[][]{
                {10, 20, 0, 0, 0, 0},
                {0, 30, 0, 40, 0, 0},
                {0, 0, 50, 60, 70, 0},
                {0, 0, 0, 0, 0, 80}};
        A = new MatrixOld(aEntries).toCsr();

        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(-1, 2));
        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(1, 145));
        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(0, -145));
    }


    @Test
    void colSwapTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0},
                {0,  30, 0,  40, 0,  0},
                {0,  0,  50, 60, 70, 0},
                {0,  0,  0,  0,  0,  80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {0,  20, 0,  10, 0,  0},
                {40, 30, 0,  0,  0,  0},
                {60, 0,  50, 0,  70, 0},
                {0,  0,  0,  0,  0,  80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 3));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0},
                {0,  30, 0,  40, 0,  0},
                {0,  0,  50, 60, 70, 0},
                {0,  0,  0,  0,  0,  80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {0,  20, 0,  10, 0,  0},
                {40, 30, 0,  0,  0,  0},
                {60, 0,  50, 0,  70, 0},
                {0,  0,  0,  0,  0,  80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(3, 0));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0},
                {0,  30, 0,  40, 0,  0},
                {0,  0,  50, 60, 70, 0},
                {0,  0,  0,  0,  0,  80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {10, 0,  0,  20, 0,  0},
                {0,  40, 0,  30, 0,  0},
                {0,  60, 50, 0,  70, 0},
                {0,  0,  0,  0,  0,  80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(1, 3));

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0 },
                {0,  30, 0,  40, 0,  0 },
                {0,  0,  50, 60, 70, 0 },
                {0,  0,  0,  0,  0,  80}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {0,  20, 0,  0,  0,  10},
                {0,  30, 0,  40, 0,  0 },
                {0,  0,  50, 60, 70, 0 },
                {80, 0,  0,  0,  0,  0 }};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 5));

        // ---------------------- Sub-case 5 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0 },
                {0,  30, 0,  0,  0,  0 },
                {0,  40, 50, 60, 70, 0 },
                {0,  0,  0,  80,  0, 90}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {10, 0,  0,  20, 0,  0 },
                {0,  0,  0,  30, 0,  0 },
                {0,  60, 50, 40, 70, 0 },
                {0,  80, 0,  0,  0,  90}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(1, 3));

        // ---------------------- Sub-case 6 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0 },
                {0,  30, 0,  0,  0,  0 },
                {0,  40, 50, 60, 70, 0 },
                {0,  0,  0,  80,  0, 90}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {10, 20, 0,  0,  0,  0 },
                {0,  30, 0,  0,  0,  0 },
                {0,  40, 50, 0,  70, 60 },
                {0,  0,  0,  90, 0, 80}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(3, 5));

        // ---------------------- Sub-case 7 ----------------------
        aEntries = new double[][]{
                {10, 20, 0,  0,  0,  0 },
                {0,  30, 0,  0,  0,  0 },
                {0,  40, 50, 60, 70, 0 },
                {0,  0,  0,  80,  0, 90}};
        A = new MatrixOld(aEntries).toCsr();
        expEntries = new double[][]{
                {20, 10, 0,  0,  0,  0 },
                {30, 0,  0,  0,  0,  0 },
                {40, 0,  50, 60, 70, 0 },
                {0,  0,  0,  80,  0, 90}};
        exp = new MatrixOld(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 1));
    }
}
