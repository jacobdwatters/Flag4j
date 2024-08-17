package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixSumRowsColsTests {
    double[][] aEntries;
    double[] expEntries;
    MatrixOld A;
    VectorOld exp;

    @Test
    void sumRowsTestCase() {
        // ---------------- Sub-case 1 ----------------
        aEntries = new double[][]
                {{1.5468, -9.234, 10.2},
                {8.234, 7.1, 3},
                {-999.135, 7982, 1.78}};
        A = new MatrixOld(aEntries);
        expEntries = new double[]{1.5468+8.234-999.135, -9.234+7.1+7982, 10.2+3+1.78};
        exp = new VectorOld(expEntries);

        assertEquals(exp, A.sumRows());

        // ---------------- Sub-case 2 ----------------
        aEntries = new double[][]
                {{1.5468, -9.234, 10.2}};
        A = new MatrixOld(aEntries);
        expEntries = new double[]{1.5468, -9.234, 10.2};
        exp = new VectorOld(expEntries);

        assertEquals(exp, A.sumRows());
    }


    @Test
    void sumColsTestCase() {
        // ---------------- Sub-case 1 ----------------
        aEntries = new double[][]
                {{1.5468, -9.234, 10.2},
                        {8.234, 7.1, 3},
                        {-999.135, 7982, 1.78}};
        A = new MatrixOld(aEntries);
        expEntries = new double[]{1.5468-9.234+10.2, 8.234+7.1+3, -999.135+7982+1.78};
        exp = new VectorOld(expEntries);

        assertEquals(exp, A.sumCols());

        // ---------------- Sub-case 2 ----------------
        aEntries = new double[][]{{1.5468}, {8.234}, {-999.135}};
        A = new MatrixOld(aEntries);
        expEntries = new double[]{1.5468, 8.234, -999.135};
        exp = new VectorOld(expEntries);

        assertEquals(exp, A.sumCols());
    }
}
