package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixAddToEachRowColTest {
    static CsrMatrix A;
    static double[][] aEntries;

    static VectorOld bReDe;
    static CVectorOld bCmDe;
    static CooVector bReSp;
    static CooCVector bCmSp;
    static double[] bRealEntries;
    static CNumber[] bCmpEntries;

    static MatrixOld expReal;
    static double[][] expRealEntries;

    static CMatrixOld expCmp;
    static CNumber[][] expCmpEntries;


    @Test
    void realColAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReDe = new VectorOld(bRealEntries);
        expRealEntries = new double[][]{
                {1.4, 1.34+1.4, 1.4, 1.4, -0.24+1.4, 1.4, 1.4, 2.999184+1.4},
                {1.459903-0.2424, -0.2424, -0.2424, -0.2424, -0.2424, -0.2424, 345.14-0.2424, -0.2424},
                {10024, 10024, 10024, 14.329+10024, 10024, 10024, 9144.4+10024, 10024},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {1.45, 1.45, 1.45, 1.45, 1.45, 1.14+1.45, 2055.2+1.45, 9.435+1.45}
        };
        expReal = new MatrixOld(expRealEntries);

        assertEquals(expReal, A.addToEachCol(bReDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReDe = new VectorOld(bRealEntries);
        expRealEntries = new double[][]{
                {0+0.234, 3.4+0.234, 0+0.234, 0+0.234},
                {1.415-0.00204, -0.0024-0.00204, 105.2-0.00204, -3.14-0.00204},
                {0+100.14, 0+100.14, 0+100.14, 0+100.14},
                {0-9345.23, 0-9345.23, -9.324-9345.23, 0-9345.23},
                {14.52+1, 1, 23.4+1, 1},
                {0+0.2525, 0+0.2525, 500.1+0.2525, 0+0.2525},
                {20.4+4356.7, 0+4356.7, 0+4356.7, 0+4356.7},
                {0-99999.1, 0-99999.1, 0-99999.1, 145.5-99999.1},
        };
        expReal = new MatrixOld(expRealEntries);

        assertEquals(expReal, A.addToEachCol(bReDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525};
        bReDe = new VectorOld(bRealEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReDe));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReDe = new VectorOld(bRealEntries);
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReDe));
    }
    

    @Test
    void realRowAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReDe = new VectorOld(bRealEntries);
        expRealEntries = new double[][]{
                {1.4, 1.34+1.4, 1.4, 1.4, -0.24+1.4, 1.4, 1.4, 2.999184+1.4},
                {1.459903-0.2424, -0.2424, -0.2424, -0.2424, -0.2424, -0.2424, 345.14-0.2424, -0.2424},
                {10024, 10024, 10024, 14.329+10024, 10024, 10024, 9144.4+10024, 10024},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {1.45, 1.45, 1.45, 1.45, 1.45, 1.14+1.45, 2055.2+1.45, 9.435+1.45}
        };
        expReal = new MatrixOld(expRealEntries).T();

        assertEquals(expReal, A.addToEachRow(bReDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReDe = new VectorOld(bRealEntries);
        expRealEntries = new double[][]{
                {0+0.234, 3.4+0.234, 0+0.234, 0+0.234},
                {1.415-0.00204, -0.0024-0.00204, 105.2-0.00204, -3.14-0.00204},
                {0+100.14, 0+100.14, 0+100.14, 0+100.14},
                {0-9345.23, 0-9345.23, -9.324-9345.23, 0-9345.23},
                {14.52+1, 1, 23.4+1, 1},
                {0+0.2525, 0+0.2525, 500.1+0.2525, 0+0.2525},
                {20.4+4356.7, 0+4356.7, 0+4356.7, 0+4356.7},
                {0-99999.1, 0-99999.1, 0-99999.1, 145.5-99999.1},
        };
        expReal = new MatrixOld(expRealEntries).T();

        assertEquals(expReal, A.addToEachRow(bReDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14};
        bReDe = new VectorOld(bRealEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReDe));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReDe = new VectorOld(bRealEntries);
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReDe));
    }


    @Test
    void complexColAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24), 
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmDe = new CVectorOld(bCmpEntries);
        expCmpEntries = new CNumber[][]{
                {new CNumber(0.24+0, 1.235), new CNumber(0.24+1.34, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+-0.24, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+2.999184, 1.235)},
                {new CNumber(-100.24+1.459903), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+345.14), new CNumber(-100.24+0)},
                {new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(14.329, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(9144.4, 15.2), new CNumber(0, 15.2)},
                {new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+9.41, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1)},
                {new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+1.14, -75.243), new CNumber(1.52+2055.2, -75.243), new CNumber(1.52+9.435, -75.243)}
        };
        expCmp = new CMatrixOld(expCmpEntries);

        assertEquals(expCmp, A.addToEachCol(bCmDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmDe = new CVectorOld(bCmpEntries);
        expCmpEntries = new CNumber[][]{
                {new CNumber(14, 35.3), new CNumber(14+3.4, 35.3), new CNumber(14, 35.3), new CNumber(14, 35.3)},
                {new CNumber(-0.452+1.415, 25.1), new CNumber(-0.452+-0.0024, 25.1), new CNumber(-0.452+105.2, 25.1), new CNumber(-0.452+-3.14, 25.1)},
                {new CNumber(9834), new CNumber(9834), new CNumber(9834), new CNumber(9834)},
                {new CNumber(0, 345.1), new CNumber(0, 345.1), new CNumber(-9.324, 345.1), new CNumber(0, 345.1)},
                {new CNumber(9.435+14.52, 14.3), new CNumber(9.435+0, 14.3), new CNumber(9.435+23.4, 14.3), new CNumber(9.435+0, 14.3)},
                {new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+500.1, -92.4), new CNumber(-0.35345+0, -92.4)},
                {new CNumber(3405.1+20.4), new CNumber(3405.1+0), new CNumber(3405.1+0), new CNumber(3405.1+0)},
                {new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(145.5, 7510)},
        };
        expCmp = new CMatrixOld(expCmpEntries);

        assertEquals(expCmp, A.addToEachCol(bCmDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525};
        bReDe = new VectorOld(bRealEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReDe));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReDe = new VectorOld(bRealEntries);
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReDe));
    }


    @Test
    void complexRowAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmDe = new CVectorOld(bCmpEntries);
        expCmpEntries = new CNumber[][]{
                {new CNumber(0.24+0, 1.235), new CNumber(0.24+1.34, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+-0.24, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+2.999184, 1.235)},
                {new CNumber(-100.24+1.459903), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+345.14), new CNumber(-100.24+0)},
                {new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(14.329, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(9144.4, 15.2), new CNumber(0, 15.2)},
                {new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+9.41, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1)},
                {new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+1.14, -75.243), new CNumber(1.52+2055.2, -75.243), new CNumber(1.52+9.435, -75.243)}
        };
        expCmp = new CMatrixOld(expCmpEntries).T();

        assertEquals(expCmp, A.addToEachRow(bCmDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmDe = new CVectorOld(bCmpEntries);
        expCmpEntries = new CNumber[][]{
                {new CNumber(14, 35.3), new CNumber(14+3.4, 35.3), new CNumber(14, 35.3), new CNumber(14, 35.3)},
                {new CNumber(-0.452+1.415, 25.1), new CNumber(-0.452+-0.0024, 25.1), new CNumber(-0.452+105.2, 25.1), new CNumber(-0.452+-3.14, 25.1)},
                {new CNumber(9834), new CNumber(9834), new CNumber(9834), new CNumber(9834)},
                {new CNumber(0, 345.1), new CNumber(0, 345.1), new CNumber(-9.324, 345.1), new CNumber(0, 345.1)},
                {new CNumber(9.435+14.52, 14.3), new CNumber(9.435+0, 14.3), new CNumber(9.435+23.4, 14.3), new CNumber(9.435+0, 14.3)},
                {new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+500.1, -92.4), new CNumber(-0.35345+0, -92.4)},
                {new CNumber(3405.1+20.4), new CNumber(3405.1+0), new CNumber(3405.1+0), new CNumber(3405.1+0)},
                {new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(145.5, 7510)},
        };
        expCmp = new CMatrixOld(expCmpEntries).T();

        assertEquals(expCmp, A.addToEachRow(bCmDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14};
        bReDe = new VectorOld(bRealEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReDe));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReDe = new VectorOld(bRealEntries);
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReDe));
    }

    // --------------------------------------------- Sparse Below --------------------------------------------
    
    @Test
    void realSpColAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expRealEntries = new double[][]{
                {1.4, 1.34+1.4, 1.4, 1.4, -0.24+1.4, 1.4, 1.4, 2.999184+1.4},
                {1.459903-0.2424, -0.2424, -0.2424, -0.2424, -0.2424, -0.2424, 345.14-0.2424, -0.2424},
                {10024, 10024, 10024, 14.329+10024, 10024, 10024, 9144.4+10024, 10024},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {1.45, 1.45, 1.45, 1.45, 1.45, 1.14+1.45, 2055.2+1.45, 9.435+1.45}
        };
        expReal = new MatrixOld(expRealEntries);

        assertEquals(expReal, A.addToEachCol(bReSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expRealEntries = new double[][]{
                {0+0.234, 3.4+0.234, 0+0.234, 0+0.234},
                {1.415-0.00204, -0.0024-0.00204, 105.2-0.00204, -3.14-0.00204},
                {0+100.14, 0+100.14, 0+100.14, 0+100.14},
                {0-9345.23, 0-9345.23, -9.324-9345.23, 0-9345.23},
                {14.52+1, 1, 23.4+1, 1},
                {0+0.2525, 0+0.2525, 500.1+0.2525, 0+0.2525},
                {20.4+4356.7, 0+4356.7, 0+4356.7, 0+4356.7},
                {0-99999.1, 0-99999.1, 0-99999.1, 145.5-99999.1},
        };
        expReal = new MatrixOld(expRealEntries);

        assertEquals(expReal, A.addToEachCol(bReSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525};
        bReSp = new VectorOld(bRealEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReSp));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReSp = new VectorOld(bRealEntries).toCoo();
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReSp));
    }


    @Test
    void realSpRowAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expRealEntries = new double[][]{
                {1.4, 1.34+1.4, 1.4, 1.4, -0.24+1.4, 1.4, 1.4, 2.999184+1.4},
                {1.459903-0.2424, -0.2424, -0.2424, -0.2424, -0.2424, -0.2424, 345.14-0.2424, -0.2424},
                {10024, 10024, 10024, 14.329+10024, 10024, 10024, 9144.4+10024, 10024},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {1.45, 1.45, 1.45, 1.45, 1.45, 1.14+1.45, 2055.2+1.45, 9.435+1.45}
        };
        expReal = new MatrixOld(expRealEntries).T();

        assertEquals(expReal, A.addToEachRow(bReSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expRealEntries = new double[][]{
                {0+0.234, 3.4+0.234, 0+0.234, 0+0.234},
                {1.415-0.00204, -0.0024-0.00204, 105.2-0.00204, -3.14-0.00204},
                {0+100.14, 0+100.14, 0+100.14, 0+100.14},
                {0-9345.23, 0-9345.23, -9.324-9345.23, 0-9345.23},
                {14.52+1, 1, 23.4+1, 1},
                {0+0.2525, 0+0.2525, 500.1+0.2525, 0+0.2525},
                {20.4+4356.7, 0+4356.7, 0+4356.7, 0+4356.7},
                {0-99999.1, 0-99999.1, 0-99999.1, 145.5-99999.1},
        };
        expReal = new MatrixOld(expRealEntries).T();

        assertEquals(expReal, A.addToEachRow(bReSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14};
        bReSp = new VectorOld(bRealEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReSp));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReSp = new VectorOld(bRealEntries).toCoo();
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReSp));
    }


    @Test
    void complexSpColAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expCmpEntries = new CNumber[][]{
                {new CNumber(0.24+0, 1.235), new CNumber(0.24+1.34, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+-0.24, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+2.999184, 1.235)},
                {new CNumber(-100.24+1.459903), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+345.14), new CNumber(-100.24+0)},
                {new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(14.329, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(9144.4, 15.2), new CNumber(0, 15.2)},
                {new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+9.41, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1)},
                {new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+1.14, -75.243), new CNumber(1.52+2055.2, -75.243), new CNumber(1.52+9.435, -75.243)}
        };
        expCmp = new CMatrixOld(expCmpEntries);

        assertEquals(expCmp, A.addToEachCol(bCmSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expCmpEntries = new CNumber[][]{
                {new CNumber(14, 35.3), new CNumber(14+3.4, 35.3), new CNumber(14, 35.3), new CNumber(14, 35.3)},
                {new CNumber(-0.452+1.415, 25.1), new CNumber(-0.452+-0.0024, 25.1), new CNumber(-0.452+105.2, 25.1), new CNumber(-0.452+-3.14, 25.1)},
                {new CNumber(9834), new CNumber(9834), new CNumber(9834), new CNumber(9834)},
                {new CNumber(0, 345.1), new CNumber(0, 345.1), new CNumber(-9.324, 345.1), new CNumber(0, 345.1)},
                {new CNumber(9.435+14.52, 14.3), new CNumber(9.435+0, 14.3), new CNumber(9.435+23.4, 14.3), new CNumber(9.435+0, 14.3)},
                {new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+500.1, -92.4), new CNumber(-0.35345+0, -92.4)},
                {new CNumber(3405.1+20.4), new CNumber(3405.1+0), new CNumber(3405.1+0), new CNumber(3405.1+0)},
                {new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(145.5, 7510)},
        };
        expCmp = new CMatrixOld(expCmpEntries);

        assertEquals(expCmp, A.addToEachCol(bCmSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525};
        bReSp = new VectorOld(bRealEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReSp));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReSp = new VectorOld(bRealEntries).toCoo();
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bReSp));
    }


    @Test
    void complexSpRowAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {0, 1.34, 0, 0, -0.24, 0, 0, 2.999184},
                {1.459903, 0, 0, 0, 0, 0, 345.14, 0},
                {0, 0, 0, 14.329, 0, 0, 9144.4, 0},
                {0, 9.41, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1.14, 2055.2, 9.435}
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expCmpEntries = new CNumber[][]{
                {new CNumber(0.24+0, 1.235), new CNumber(0.24+1.34, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+-0.24, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+0, 1.235), new CNumber(0.24+2.999184, 1.235)},
                {new CNumber(-100.24+1.459903), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+0), new CNumber(-100.24+345.14), new CNumber(-100.24+0)},
                {new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(14.329, 15.2), new CNumber(0, 15.2), new CNumber(0, 15.2), new CNumber(9144.4, 15.2), new CNumber(0, 15.2)},
                {new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+9.41, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1), new CNumber(-943.1+0, -9242.1)},
                {new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+0, -75.243), new CNumber(1.52+1.14, -75.243), new CNumber(1.52+2055.2, -75.243), new CNumber(1.52+9.435, -75.243)}
        };
        expCmp = new CMatrixOld(expCmpEntries).T();

        assertEquals(expCmp, A.addToEachRow(bCmSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expCmpEntries = new CNumber[][]{
                {new CNumber(14, 35.3), new CNumber(14+3.4, 35.3), new CNumber(14, 35.3), new CNumber(14, 35.3)},
                {new CNumber(-0.452+1.415, 25.1), new CNumber(-0.452+-0.0024, 25.1), new CNumber(-0.452+105.2, 25.1), new CNumber(-0.452+-3.14, 25.1)},
                {new CNumber(9834), new CNumber(9834), new CNumber(9834), new CNumber(9834)},
                {new CNumber(0, 345.1), new CNumber(0, 345.1), new CNumber(-9.324, 345.1), new CNumber(0, 345.1)},
                {new CNumber(9.435+14.52, 14.3), new CNumber(9.435+0, 14.3), new CNumber(9.435+23.4, 14.3), new CNumber(9.435+0, 14.3)},
                {new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+0, -92.4), new CNumber(-0.35345+500.1, -92.4), new CNumber(-0.35345+0, -92.4)},
                {new CNumber(3405.1+20.4), new CNumber(3405.1+0), new CNumber(3405.1+0), new CNumber(3405.1+0)},
                {new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(0, 7510), new CNumber(145.5, 7510)},
        };
        expCmp = new CMatrixOld(expCmpEntries).T();

        assertEquals(expCmp, A.addToEachRow(bCmSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {0, 3.4, 0, 0},
                {1.415, -0.0024, 105.2, -3.14},
                {0, 0, 0, 0},
                {0, 0, -9.324, 0},
                {14.52, 0, 23.4, 0},
                {0, 0, 500.1, 0},
                {20.4, 0, 0, 0},
                {0, 0, 0, 145.5},
        };
        A = new MatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14};
        bReSp = new VectorOld(bRealEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReSp));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReSp = new VectorOld(bRealEntries).toCoo();
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReSp));
    }
}
