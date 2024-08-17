package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.arrays_old.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixAddToEachRowTest {
    static CsrCMatrix A;
    static CNumber[][] aEntries;

    static VectorOld bReDe;
    static CVectorOld bCmDe;
    static CooVector bReSp;
    static CooCVector bCmSp;
    static double[] bRealEntries;
    static CNumber[] bCmpEntries;

    static CMatrixOld exp;
    static CNumber[][] expEntries;


    @Test
    void realColAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34, -9834.1), new CNumber(0),
                        new CNumber(0), new CNumber(0, -0.24), new CNumber(0),
                        new CNumber(0), new CNumber(2.999184)},
                {new CNumber(1.459903, 1.5), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(345.14), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(14.329, -9), new CNumber(0), new CNumber(0),
                        new CNumber(9144.4), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(1.14),
                        new CNumber(1.234, 2055.2), new CNumber(9.435, 16.2)}
        };
        A = new CMatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReDe = new VectorOld(bRealEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.4), new CNumber(1.34+1.4, -9834.1), new CNumber(1.4),
                        new CNumber(1.4), new CNumber(1.4, -0.24), new CNumber(1.4),
                        new CNumber(1.4), new CNumber(2.999184+1.4)},
                {new CNumber(1.459903-0.2424, 1.5), new CNumber(-0.2424), new CNumber(-0.2424),
                        new CNumber(-0.2424), new CNumber(-0.2424), new CNumber(-0.2424),
                        new CNumber(345.14-0.2424), new CNumber(-0.2424)},
                {new CNumber(10024), new CNumber(10024), new CNumber(10024),
                        new CNumber(14.329+10024, -9), new CNumber(10024), new CNumber(10024),
                        new CNumber(9144.4+10024), new CNumber(10024)},
                {new CNumber(0), new CNumber(9.41), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0)},
                {new CNumber(1.45), new CNumber(1.45), new CNumber(1.45),
                        new CNumber(1.45), new CNumber(1.45), new CNumber(1.14+1.45),
                        new CNumber(1.234+1.45, 2055.2), new CNumber(9.435+1.45, 16.2)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bReDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0, 3.4), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, -2), new CNumber(-0.0024, 1.51), new CNumber(105.2, 2), new CNumber(-3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, -2), new CNumber(0)},
                {new CNumber(14.52, -71.2), new CNumber(0), new CNumber(23.4, 8), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 6), new CNumber(0)},
                {new CNumber(20.4, 456.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 15.2)},
        };
        A = new CMatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReDe = new VectorOld(bRealEntries);
        expEntries = new CNumber[][]{
                {new CNumber(0.234), new CNumber(0.234, 3.4), new CNumber(0.234), new CNumber(0.234)},
                {new CNumber(1.415-0.00204, -2), new CNumber(-0.0024-0.00204, 1.51), new CNumber(105.2-0.00204, 2), new CNumber(-3.14-0.00204)},
                {new CNumber(100.14), new CNumber(100.14), new CNumber(100.14), new CNumber(100.14)},
                {new CNumber(-9345.23), new CNumber(-9345.23), new CNumber(-9.324-9345.23, -2), new CNumber(-9345.23)},
                {new CNumber(14.52+1, -71.2), new CNumber(1), new CNumber(23.4+1, 8), new CNumber(1)},
                {new CNumber(0.2525), new CNumber(0.2525), new CNumber(500.1+0.2525, 6), new CNumber(0.2525)},
                {new CNumber(20.4+4356.7, 456.1), new CNumber(4356.7), new CNumber(4356.7), new CNumber(4356.7)},
                {new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(145.5-99999.1, 15.2)},
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bReDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0, 3.4), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, -2), new CNumber(-0.0024, 1.51), new CNumber(105.2, 2), new CNumber(-3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, -2), new CNumber(0)},
                {new CNumber(14.52, -71.2), new CNumber(0), new CNumber(23.4, 8), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 6), new CNumber(0)},
                {new CNumber(20.4, 456.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 15.2)},
        };
        A = new CMatrixOld(aEntries).toCsr();
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
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34, -9.3), new CNumber(0), new CNumber(0),
                        new CNumber(-0.24, 20.3), new CNumber(0), new CNumber(0), new CNumber(0, 2.999184)},
                {new CNumber(1.459903), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(345.14, 12.5), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(14.329,1),
                        new CNumber(0), new CNumber(0), new CNumber(9144.4, 2), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41, -50), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(1.14, 15.54), new CNumber(2055.2, 1), new CNumber(9.435, 2)}
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReDe = new VectorOld(bRealEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.4), new CNumber(1.34+1.4, -9.3), new CNumber(1.4), new CNumber(1.4),
                        new CNumber(-0.24+1.4, 20.3), new CNumber(1.4), new CNumber(1.4), new CNumber(1.4, 2.999184)},
                {new CNumber(1.459903-0.2424), new CNumber(-0.2424), new CNumber(-0.2424), new CNumber(-0.2424),
                        new CNumber(-0.2424), new CNumber(-0.2424), new CNumber(345.14-0.2424, 12.5), new CNumber(-0.2424)},
                {new CNumber(10024), new CNumber(10024), new CNumber(10024), new CNumber(14.329+10024,1),
                        new CNumber(10024), new CNumber(10024), new CNumber(9144.4+10024, 2), new CNumber(10024)},
                {new CNumber(0), new CNumber(9.41, -50), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(1.45), new CNumber(1.45), new CNumber(1.45), new CNumber(1.45),
                        new CNumber(1.45), new CNumber(1.14+1.45, 15.54), new CNumber(2055.2+1.45, 1), new CNumber(9.435+1.45, 2)}
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bReDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.345), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415), new CNumber(-0.0024), new CNumber(105.2, 70.12), new CNumber(-3.14, 0.002)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 1), new CNumber(0), new CNumber(23.4, -92.1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, -31.5), new CNumber(0)},
                {new CNumber(20.4, 31), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 93.1)},
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReDe = new VectorOld(bRealEntries);
        expEntries = new CNumber[][]{
                {new CNumber(0.234), new CNumber(3.4+0.234, 1.345), new CNumber(0.234), new CNumber(0.234)},
                {new CNumber(1.415-0.00204), new CNumber(-0.0024-0.00204), new CNumber(105.2-0.00204, 70.12), new CNumber(-3.14-0.00204, 0.002)},
                {new CNumber(100.14), new CNumber(100.14), new CNumber(100.14), new CNumber(100.14)},
                {new CNumber(-9345.23), new CNumber(-9345.23), new CNumber(-9.324-9345.23), new CNumber(-9345.23)},
                {new CNumber(14.52+1, 1), new CNumber(1), new CNumber(23.4+1, -92.1), new CNumber(1)},
                {new CNumber(0.2525), new CNumber(0.2525), new CNumber(500.1+0.2525,-31.5), new CNumber(0.2525)},
                {new CNumber(20.4+4356.7, 31), new CNumber(4356.7), new CNumber(4356.7), new CNumber(4356.7)},
                {new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(145.5-99999.1, 93.1)},
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bReDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.345), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415), new CNumber(-0.0024), new CNumber(105.2, 70.12), new CNumber(-3.14, 0.002)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 1), new CNumber(0), new CNumber(23.4, -92.1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1), new CNumber(0)},
                {new CNumber(20.4, 31), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 93.1)},
        };
        A = new CMatrixOld(aEntries).toCsr();
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
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34, 25.26), new CNumber(0), new CNumber(0),
                        new CNumber(-0.24, 2.6), new CNumber(0), new CNumber(0), new CNumber(2.999184, -88.4)},
                {new CNumber(1.459903, 236.2), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(345.14, -3.1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(14.329, 15.6),
                        new CNumber(0), new CNumber(0), new CNumber(9144.4, 3.3), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41, -2), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(1.14, 3), new CNumber(0, 2055.2), new CNumber(9.435)}
        };
        A = new CMatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmDe = new CVectorOld(bCmpEntries);
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(1.34, 25.26).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]),
                        new CNumber(-0.24, 2.6).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(2.999184, -88.4).add(bCmpEntries[0])},
                {new CNumber(1.459903, 236.2).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]),
                        new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(345.14, -3.1).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(14.329, 15.6).add(bCmpEntries[2]),
                        new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(9144.4, 3.3).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(9.41, -2).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]),
                        new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]),
                        new CNumber(0).add(bCmpEntries[4]), new CNumber(1.14, 3).add(bCmpEntries[4]), new CNumber(0, 2055.2).add(bCmpEntries[4]), new CNumber(9.435).add(bCmpEntries[4])}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bCmDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, -82.1), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.2), new CNumber(-0.0024, 2356.12), new CNumber(105.2, -1), new CNumber(0, -3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, 12), new CNumber(0)},
                {new CNumber(14.52, -8.1), new CNumber(0), new CNumber(23.4, 602.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 302.5), new CNumber(0)},
                {new CNumber(20.4, 1.25), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 2.6)},
        };
        A = new CMatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmDe = new CVectorOld(bCmpEntries);
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(3.4, -82.1).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0])},
                {new CNumber(1.415, 16.2).add(bCmpEntries[1]), new CNumber(-0.0024, 2356.12).add(bCmpEntries[1]), new CNumber(105.2, -1).add(bCmpEntries[1]), new CNumber(0, -3.14).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(-9.324, 12).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(14.52, -8.1).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(23.4, 602.2).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4])},
                {new CNumber(0).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5]), new CNumber(500.1, 302.5).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5])},
                {new CNumber(20.4, 1.25).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6])},
                {new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(145.5, 2.6).add(bCmpEntries[7])},
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bCmDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, -82.1), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.2), new CNumber(-0.0024, 2356.12), new CNumber(105.2, -1), new CNumber(0, -3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, 12), new CNumber(0)},
                {new CNumber(14.52, -8.1), new CNumber(0), new CNumber(23.4, 602.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 302.5), new CNumber(0)},
                {new CNumber(20.4, 1.25), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 2.6)},
        };
        A = new CMatrixOld(aEntries).toCsr();
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
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34), new CNumber(0), new CNumber(0),
                        new CNumber(-0.24), new CNumber(0), new CNumber(0), new CNumber(2.999184)},
                {new CNumber(1.459903), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(345.14), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(14.329),
                        new CNumber(0), new CNumber(0), new CNumber(9144.4), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(1.14), new CNumber(2055.2), new CNumber(9.435)}
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmDe = new CVectorOld(bCmpEntries);
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(1.34).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]),
                        new CNumber(-0.24).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(2.999184).add(bCmpEntries[0])},
                {new CNumber(1.459903).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]),
                        new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(345.14).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(14.329).add(bCmpEntries[2]),
                        new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(9144.4).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(9.41).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]),
                        new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]),
                        new CNumber(0).add(bCmpEntries[4]), new CNumber(1.14).add(bCmpEntries[4]), new CNumber(2055.2).add(bCmpEntries[4]), new CNumber(9.435).add(bCmpEntries[4])}
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bCmDe));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.5), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.1), new CNumber(-0.0024, 25), new CNumber(105.2, 0.0015), new CNumber(-3.14, 801.2)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 50.1), new CNumber(0), new CNumber(23.4, -9993469.251), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 1.566), new CNumber(0)},
                {new CNumber(20.4, 85781.2), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 145.5)},
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmDe = new CVectorOld(bCmpEntries);
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(3.4, 1.5).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0])},
                {new CNumber(1.415, 16.1).add(bCmpEntries[1]), new CNumber(-0.0024, 25).add(bCmpEntries[1]), new CNumber(105.2, 0.0015).add(bCmpEntries[1]), new CNumber(-3.14, 801.2).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(-9.324).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(14.52, 50.1).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(23.4, -9993469.251).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4])},
                {new CNumber(0).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5]), new CNumber(500.1, 1.566).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5])},
                {new CNumber(20.4, 85781.2).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6])},
                {new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(0, 145.5).add(bCmpEntries[7])},
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bCmDe));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.5), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.1), new CNumber(-0.0024, 25), new CNumber(105.2, 0.0015), new CNumber(-3.14, 801.2)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 50.1), new CNumber(0), new CNumber(23.4, -9993469.251), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 1.566), new CNumber(0)},
                {new CNumber(20.4, 85781.2), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 145.5)},
        };
        A = new CMatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14};
        bReDe = new VectorOld(bRealEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReDe));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReDe = new VectorOld(bRealEntries);
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReDe));
    }


    // ------------------------------------ Sparse Below ------------------------------------

    @Test
    void realSpColAddTests() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34, -9834.1), new CNumber(0),
                        new CNumber(0), new CNumber(0, -0.24), new CNumber(0),
                        new CNumber(0), new CNumber(2.999184)},
                {new CNumber(1.459903, 1.5), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(345.14), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(14.329, -9), new CNumber(0), new CNumber(0),
                        new CNumber(9144.4), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(1.14),
                        new CNumber(1.234, 2055.2), new CNumber(9.435, 16.2)}
        };
        A = new CMatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(1.4), new CNumber(1.34+1.4, -9834.1), new CNumber(1.4),
                        new CNumber(1.4), new CNumber(1.4, -0.24), new CNumber(1.4),
                        new CNumber(1.4), new CNumber(2.999184+1.4)},
                {new CNumber(1.459903-0.2424, 1.5), new CNumber(-0.2424), new CNumber(-0.2424),
                        new CNumber(-0.2424), new CNumber(-0.2424), new CNumber(-0.2424),
                        new CNumber(345.14-0.2424), new CNumber(-0.2424)},
                {new CNumber(10024), new CNumber(10024), new CNumber(10024),
                        new CNumber(14.329+10024, -9), new CNumber(10024), new CNumber(10024),
                        new CNumber(9144.4+10024), new CNumber(10024)},
                {new CNumber(0), new CNumber(9.41), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0)},
                {new CNumber(1.45), new CNumber(1.45), new CNumber(1.45),
                        new CNumber(1.45), new CNumber(1.45), new CNumber(1.14+1.45),
                        new CNumber(1.234+1.45, 2055.2), new CNumber(9.435+1.45, 16.2)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bReSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0, 3.4), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, -2), new CNumber(-0.0024, 1.51), new CNumber(105.2, 2), new CNumber(-3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, -2), new CNumber(0)},
                {new CNumber(14.52, -71.2), new CNumber(0), new CNumber(23.4, 8), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 6), new CNumber(0)},
                {new CNumber(20.4, 456.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 15.2)},
        };
        A = new CMatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0.234), new CNumber(0.234, 3.4), new CNumber(0.234), new CNumber(0.234)},
                {new CNumber(1.415-0.00204, -2), new CNumber(-0.0024-0.00204, 1.51), new CNumber(105.2-0.00204, 2), new CNumber(-3.14-0.00204)},
                {new CNumber(100.14), new CNumber(100.14), new CNumber(100.14), new CNumber(100.14)},
                {new CNumber(-9345.23), new CNumber(-9345.23), new CNumber(-9.324-9345.23, -2), new CNumber(-9345.23)},
                {new CNumber(14.52+1, -71.2), new CNumber(1), new CNumber(23.4+1, 8), new CNumber(1)},
                {new CNumber(0.2525), new CNumber(0.2525), new CNumber(500.1+0.2525, 6), new CNumber(0.2525)},
                {new CNumber(20.4+4356.7, 456.1), new CNumber(4356.7), new CNumber(4356.7), new CNumber(4356.7)},
                {new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(145.5-99999.1, 15.2)},
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bReSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0, 3.4), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, -2), new CNumber(-0.0024, 1.51), new CNumber(105.2, 2), new CNumber(-3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, -2), new CNumber(0)},
                {new CNumber(14.52, -71.2), new CNumber(0), new CNumber(23.4, 8), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 6), new CNumber(0)},
                {new CNumber(20.4, 456.1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 15.2)},
        };
        A = new CMatrixOld(aEntries).toCsr();
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
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34, -9.3), new CNumber(0), new CNumber(0),
                        new CNumber(-0.24, 20.3), new CNumber(0), new CNumber(0), new CNumber(0, 2.999184)},
                {new CNumber(1.459903), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(345.14, 12.5), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(14.329,1),
                        new CNumber(0), new CNumber(0), new CNumber(9144.4, 2), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41, -50), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(1.14, 15.54), new CNumber(2055.2, 1), new CNumber(9.435, 2)}
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{1.4, -0.2424, 10024, 0, 1.45};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(1.4), new CNumber(1.34+1.4, -9.3), new CNumber(1.4), new CNumber(1.4),
                        new CNumber(-0.24+1.4, 20.3), new CNumber(1.4), new CNumber(1.4), new CNumber(1.4, 2.999184)},
                {new CNumber(1.459903-0.2424), new CNumber(-0.2424), new CNumber(-0.2424), new CNumber(-0.2424),
                        new CNumber(-0.2424), new CNumber(-0.2424), new CNumber(345.14-0.2424, 12.5), new CNumber(-0.2424)},
                {new CNumber(10024), new CNumber(10024), new CNumber(10024), new CNumber(14.329+10024,1),
                        new CNumber(10024), new CNumber(10024), new CNumber(9144.4+10024, 2), new CNumber(10024)},
                {new CNumber(0), new CNumber(9.41, -50), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(1.45), new CNumber(1.45), new CNumber(1.45), new CNumber(1.45),
                        new CNumber(1.45), new CNumber(1.14+1.45, 15.54), new CNumber(2055.2+1.45, 1), new CNumber(9.435+1.45, 2)}
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bReSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.345), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415), new CNumber(-0.0024), new CNumber(105.2, 70.12), new CNumber(-3.14, 0.002)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 1), new CNumber(0), new CNumber(23.4, -92.1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, -31.5), new CNumber(0)},
                {new CNumber(20.4, 31), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 93.1)},
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 4356.7, -99999.1};
        bReSp = new VectorOld(bRealEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0.234), new CNumber(3.4+0.234, 1.345), new CNumber(0.234), new CNumber(0.234)},
                {new CNumber(1.415-0.00204), new CNumber(-0.0024-0.00204), new CNumber(105.2-0.00204, 70.12), new CNumber(-3.14-0.00204, 0.002)},
                {new CNumber(100.14), new CNumber(100.14), new CNumber(100.14), new CNumber(100.14)},
                {new CNumber(-9345.23), new CNumber(-9345.23), new CNumber(-9.324-9345.23), new CNumber(-9345.23)},
                {new CNumber(14.52+1, 1), new CNumber(1), new CNumber(23.4+1, -92.1), new CNumber(1)},
                {new CNumber(0.2525), new CNumber(0.2525), new CNumber(500.1+0.2525,-31.5), new CNumber(0.2525)},
                {new CNumber(20.4+4356.7, 31), new CNumber(4356.7), new CNumber(4356.7), new CNumber(4356.7)},
                {new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(-99999.1), new CNumber(145.5-99999.1, 93.1)},
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bReSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.345), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415), new CNumber(-0.0024), new CNumber(105.2, 70.12), new CNumber(-3.14, 0.002)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 1), new CNumber(0), new CNumber(23.4, -92.1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1), new CNumber(0)},
                {new CNumber(20.4, 31), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 93.1)},
        };
        A = new CMatrixOld(aEntries).toCsr();
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
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34, 25.26), new CNumber(0), new CNumber(0),
                        new CNumber(-0.24, 2.6), new CNumber(0), new CNumber(0), new CNumber(2.999184, -88.4)},
                {new CNumber(1.459903, 236.2), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(345.14, -3.1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(14.329, 15.6),
                        new CNumber(0), new CNumber(0), new CNumber(9144.4, 3.3), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41, -2), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(1.14, 3), new CNumber(0, 2055.2), new CNumber(9.435)}
        };
        A = new CMatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(1.34, 25.26).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]),
                        new CNumber(-0.24, 2.6).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(2.999184, -88.4).add(bCmpEntries[0])},
                {new CNumber(1.459903, 236.2).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]),
                        new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(345.14, -3.1).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(14.329, 15.6).add(bCmpEntries[2]),
                        new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(9144.4, 3.3).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(9.41, -2).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]),
                        new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]),
                        new CNumber(0).add(bCmpEntries[4]), new CNumber(1.14, 3).add(bCmpEntries[4]), new CNumber(0, 2055.2).add(bCmpEntries[4]), new CNumber(9.435).add(bCmpEntries[4])}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bCmSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, -82.1), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.2), new CNumber(-0.0024, 2356.12), new CNumber(105.2, -1), new CNumber(0, -3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, 12), new CNumber(0)},
                {new CNumber(14.52, -8.1), new CNumber(0), new CNumber(23.4, 602.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 302.5), new CNumber(0)},
                {new CNumber(20.4, 1.25), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 2.6)},
        };
        A = new CMatrixOld(aEntries).toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(3.4, -82.1).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0])},
                {new CNumber(1.415, 16.2).add(bCmpEntries[1]), new CNumber(-0.0024, 2356.12).add(bCmpEntries[1]), new CNumber(105.2, -1).add(bCmpEntries[1]), new CNumber(0, -3.14).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(-9.324, 12).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(14.52, -8.1).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(23.4, 602.2).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4])},
                {new CNumber(0).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5]), new CNumber(500.1, 302.5).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5])},
                {new CNumber(20.4, 1.25).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6])},
                {new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(145.5, 2.6).add(bCmpEntries[7])},
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bCmSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, -82.1), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.2), new CNumber(-0.0024, 2356.12), new CNumber(105.2, -1), new CNumber(0, -3.14)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324, 12), new CNumber(0)},
                {new CNumber(14.52, -8.1), new CNumber(0), new CNumber(23.4, 602.2), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 302.5), new CNumber(0)},
                {new CNumber(20.4, 1.25), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.5, 2.6)},
        };
        A = new CMatrixOld(aEntries).toCsr();
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
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1.34), new CNumber(0), new CNumber(0),
                        new CNumber(-0.24), new CNumber(0), new CNumber(0), new CNumber(2.999184)},
                {new CNumber(1.459903), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(345.14), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(14.329),
                        new CNumber(0), new CNumber(0), new CNumber(9144.4), new CNumber(0)},
                {new CNumber(0), new CNumber(9.41), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(1.14), new CNumber(2055.2), new CNumber(9.435)}
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{new CNumber(0.24, 1.235), new CNumber(-100.24),
                new CNumber(0, 15.2), new CNumber(-943.1, -9242.1), new CNumber(1.52, -75.243)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(1.34).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]),
                        new CNumber(-0.24).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(2.999184).add(bCmpEntries[0])},
                {new CNumber(1.459903).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]),
                        new CNumber(0).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1]), new CNumber(345.14).add(bCmpEntries[1]), new CNumber(0).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(14.329).add(bCmpEntries[2]),
                        new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(9144.4).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(9.41).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]),
                        new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]),
                        new CNumber(0).add(bCmpEntries[4]), new CNumber(1.14).add(bCmpEntries[4]), new CNumber(2055.2).add(bCmpEntries[4]), new CNumber(9.435).add(bCmpEntries[4])}
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bCmSp));

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.5), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.1), new CNumber(-0.0024, 25), new CNumber(105.2, 0.0015), new CNumber(-3.14, 801.2)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 50.1), new CNumber(0), new CNumber(23.4, -9993469.251), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 1.566), new CNumber(0)},
                {new CNumber(20.4, 85781.2), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 145.5)},
        };
        A = new CMatrixOld(aEntries).T().toCsr();
        bCmpEntries = new CNumber[]{
                new CNumber(14, 35.3), new CNumber(-0.452, 25.1), new CNumber(9834), new CNumber(0, 345.1),
                new CNumber(9.435, 14.3), new CNumber(-0.35345, -92.4), new CNumber(3405.1), new CNumber(0, 7510)};
        bCmSp = new CVectorOld(bCmpEntries).toCoo();
        expEntries = new CNumber[][]{
                {new CNumber(0).add(bCmpEntries[0]), new CNumber(3.4, 1.5).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0]), new CNumber(0).add(bCmpEntries[0])},
                {new CNumber(1.415, 16.1).add(bCmpEntries[1]), new CNumber(-0.0024, 25).add(bCmpEntries[1]), new CNumber(105.2, 0.0015).add(bCmpEntries[1]), new CNumber(-3.14, 801.2).add(bCmpEntries[1])},
                {new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2]), new CNumber(0).add(bCmpEntries[2])},
                {new CNumber(0).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3]), new CNumber(-9.324).add(bCmpEntries[3]), new CNumber(0).add(bCmpEntries[3])},
                {new CNumber(14.52, 50.1).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4]), new CNumber(23.4, -9993469.251).add(bCmpEntries[4]), new CNumber(0).add(bCmpEntries[4])},
                {new CNumber(0).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5]), new CNumber(500.1, 1.566).add(bCmpEntries[5]), new CNumber(0).add(bCmpEntries[5])},
                {new CNumber(20.4, 85781.2).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6]), new CNumber(0).add(bCmpEntries[6])},
                {new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(0).add(bCmpEntries[7]), new CNumber(0, 145.5).add(bCmpEntries[7])},
        };
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, A.addToEachRow(bCmSp));

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(3.4, 1.5), new CNumber(0), new CNumber(0)},
                {new CNumber(1.415, 16.1), new CNumber(-0.0024, 25), new CNumber(105.2, 0.0015), new CNumber(-3.14, 801.2)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(-9.324), new CNumber(0)},
                {new CNumber(14.52, 50.1), new CNumber(0), new CNumber(23.4, -9993469.251), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(500.1, 1.566), new CNumber(0)},
                {new CNumber(20.4, 85781.2), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 145.5)},
        };
        A = new CMatrixOld(aEntries).toCsr();
        bRealEntries = new double[]{0.234, -0.00204, 100.14};
        bReSp = new VectorOld(bRealEntries).toCoo();

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReSp));

        // -------------------- Sub-case 4 --------------------
        bRealEntries = new double[]{0.234, -0.00204, 100.14, -9345.23, 1, 0.2525, 1, 3, 4, 5, 6, 7, 8};
        bReSp = new VectorOld(bRealEntries).toCoo();
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bReSp));
    }
}
