package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrMatrixRowColSumTests {

    static CsrMatrix A;
    static double[][] aEntriesDense;
    static VectorOld exp;
    static double[] expEntries;


    static void makeArrays() {
        A = new MatrixOld(aEntriesDense).toCsr();
        exp = new VectorOld(expEntries);
    }


    @Test
    void sumColsTests() {
        // ------------------------ sub-case 1 ------------------------
        aEntriesDense = new double[][]{
                {10, 20, 0,  0,  0,  0},
                {0,  30, 0,  40, 0,  0},
                {0,  0,  50, 60, 70, 0},
                {0,  0,  0,  0,  0, 80}};
        expEntries = new double[]{30, 70, 180, 80};
        makeArrays();
        assertEquals(exp, A.sumCols());

        // ------------------------ sub-case 2 ------------------------
        aEntriesDense = new double[1567][100];
        aEntriesDense[0][0] = 1.456;
        aEntriesDense[0][16] = -1945.6;
        aEntriesDense[5][48] = 580084.1;
        aEntriesDense[5][14] = 2;
        aEntriesDense[108][4] = 15;
        aEntriesDense[295][4] = -2.3456;
        aEntriesDense[1502][25] = 1;
        aEntriesDense[1502][26] = -24;
        aEntriesDense[1502][99] = 25901.24345;

        expEntries = new double[1567];
        expEntries[0] = 1.456 - 1945.6;
        expEntries[5] = 580084.1 + 2;
        expEntries[108] = 15;
        expEntries[295] = -2.3456;
        expEntries[1502] = 1 - 24 + 25901.24345;
        makeArrays();

        assertEquals(exp, A.sumCols());
    }


    @Test
    void sumRowsTests() {
        // ------------------------ sub-case 1 ------------------------
        aEntriesDense = new double[][]{
                {10, 20, 0,  0,  0,  0},
                {0,  30, 0,  40, 0,  0},
                {0,  0,  50, 60, 70, 0},
                {0,  0,  0,  0,  0, 80}};
        expEntries = new double[]{10, 50, 50, 100, 70, 80};
        makeArrays();
        assertEquals(exp, A.sumRows());

        // ------------------------ sub-case 2 ------------------------
        aEntriesDense = new double[100][1567];
        aEntriesDense[0][0] = 1.456;
        aEntriesDense[16][0] = -1945.6;
        aEntriesDense[48][5] = 580084.1;
        aEntriesDense[14][5] = 2;
        aEntriesDense[4][108] = 15;
        aEntriesDense[4][295] = -2.3456;
        aEntriesDense[25][1502] = 1;
        aEntriesDense[26][1502] = -24;
        aEntriesDense[99][1502] = 25901.24345;

        expEntries = new double[1567];
        expEntries[0] = 1.456 - 1945.6;
        expEntries[5] = 580084.1 + 2;
        expEntries[108] = 15;
        expEntries[295] = -2.3456;
        expEntries[1502] = 1 - 24 + 25901.24345;
        makeArrays();

        assertEquals(exp, A.sumRows());
    }
}
