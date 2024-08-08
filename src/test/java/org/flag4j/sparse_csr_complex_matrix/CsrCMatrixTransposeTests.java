package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CsrCMatrixTransposeTests {

    static CsrCMatrix A;
    static CNumber[][] aEntries;
    static CsrCMatrix exp;
    static CNumber[][] expEntries;


    private static void buildFromDense() {
        A = new CMatrix(aEntries).toCsr();
        exp = new CMatrix(expEntries).toCsr();
    }

    private static void buildFromDense(boolean buildExp) {
        A = new CMatrix(aEntries).toCsr();
        if(buildExp) exp = new CMatrix(expEntries).toCsr();
    }


    @Test
    void transposeTests() {
        // -------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.89855+0.38209i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.86937+0.81929i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16951+0.68135i"), new CNumber("0.0")},
                {new CNumber("0.65696+0.92685i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"),
                        new CNumber("0.27084+0.48366i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16383+0.02238i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16657+0.97044i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.86937+0.81929i"), new CNumber("0.65696+0.92685i"), new CNumber("0.0"),
                        new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.89855+0.38209i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16657+0.97044i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16383+0.02238i"), new CNumber("0.0"),
                        new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.16951+0.68135i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"),
                        new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.27084+0.48366i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        buildFromDense();

        assertEquals(exp, A.T());

        // -------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.98031+0.659i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.55568+0.08748i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.93371+0.94355i"), new CNumber("0.0"), new CNumber("0.46316+0.37268i"), new CNumber("0.0"), new CNumber("0.0")}};
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.93371+0.94355i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.55568+0.08748i"), new CNumber("0.46316+0.37268i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.98031+0.659i"), new CNumber("0.0"), new CNumber("0.0")}};
        buildFromDense();

        assertEquals(exp, A.T());

        // -------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.35993+0.77312i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.90417+0.16239i")},
                {new CNumber("0.27902+0.20495i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.58847+0.90372i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        expEntries = new CNumber[][]{
                {new CNumber("0.35993+0.77312i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.27902+0.20495i"), new CNumber("0.58847+0.90372i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.90417+0.16239i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        buildFromDense();

        assertEquals(exp, A.T());

        // -------------------- Sub-case 4 ---------------------
        aEntries = new CNumber[140][23];
        ArrayUtils.fill(aEntries, CNumber.ZERO);
        expEntries = new CNumber[23][140];
        ArrayUtils.fill(expEntries, CNumber.ZERO);
        buildFromDense();

        assertEquals(exp, A.T());
    }


    @Test
    void hermitianTransposeTests() {
        // -------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.89855+0.38209i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.86937+0.81929i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16951+0.68135i"), new CNumber("0.0")},
                {new CNumber("0.65696+0.92685i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.27084-0.48366i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16383+0.02238i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16657+0.97044i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.86937-0.81929i"), new CNumber("0.65696-0.92685i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.89855-0.38209i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16657-0.97044i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.16383-0.02238i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.16951-0.68135i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.27084+0.48366i"), new CNumber("0.0"), new CNumber("0.0"),
                        new CNumber("0.0")}};
        buildFromDense();

        assertEquals(exp, A.H());

        // -------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.98031+0.659i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.55568+0.08748i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.93371+0.94355i"), new CNumber("0.0"), new CNumber("0.46316+0.37268i"), new CNumber("0.0"), new CNumber("0.0")}};
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.93371-0.94355i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.55568-0.08748i"), new CNumber("0.46316-0.37268i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.98031-0.659i"), new CNumber("0.0"), new CNumber("0.0")}};
        buildFromDense();

        assertEquals(exp, A.H());

        // -------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.35993+0.77312i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.90417+0.16239i")},
                {new CNumber("0.27902-0.20495i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.58847+0.90372i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        expEntries = new CNumber[][]{
                {new CNumber("0.35993-0.77312i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.27902+0.20495i"),
                        new CNumber("0.58847-0.90372i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.90417-0.16239i"), new CNumber("0.0"), new CNumber("0.0"),
                        new CNumber("0.0")}};
        buildFromDense();

        assertEquals(exp, A.H());

        // -------------------- Sub-case 4 ---------------------
        aEntries = new CNumber[140][23];
        ArrayUtils.fill(aEntries, CNumber.ZERO);
        expEntries = new CNumber[23][140];
        ArrayUtils.fill(expEntries, CNumber.ZERO);
        buildFromDense();

        assertEquals(exp, A.H());
    }


    @Test
    void isHermitianTests() {
        // -------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[5][5];
        ArrayUtils.fill(aEntries, CNumber.ZERO);
        aEntries[0][0] = new CNumber(1.5);
        aEntries[2][1] = new CNumber(2.45, 85.12);
        aEntries[1][2] = new CNumber(2.45, -85.12);
        aEntries[3][4] = new CNumber(0, -1);
        aEntries[4][3] = new CNumber(0, 1);
        aEntries[3][3] = new CNumber(9.24);
        buildFromDense(false);

        assertTrue(A.isHermitian());

        // -------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[5][6];
        ArrayUtils.fill(aEntries, CNumber.ZERO);
        aEntries[0][0] = new CNumber(1.5);
        aEntries[2][1] = new CNumber(2.45, 85.12);
        aEntries[1][2] = new CNumber(2.45, -85.12);
        aEntries[3][4] = new CNumber(0, -1);
        aEntries[4][3] = new CNumber(0, 1);
        aEntries[3][3] = new CNumber(9.24);
        buildFromDense(false);

        assertFalse(A.isHermitian());

        // -------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[415][415];
        ArrayUtils.fill(aEntries, CNumber.ZERO);
        aEntries[0][0] = new CNumber(87.35);
        aEntries[2][1] = new CNumber(9671.4, -774.1);
        aEntries[1][2] = new CNumber(9671.4, 774.1);
        aEntries[3][4] = new CNumber(-87834.2, 12.5);
        aEntries[4][3] = new CNumber(-87834.2, -12.5);
        aEntries[3][3] = new CNumber(400);
        aEntries[45][2] = new CNumber(0, 124.34346);
        aEntries[2][45] = new CNumber(0, -124.34346);
        aEntries[13][44] = new CNumber(-93.15, 15.12355);
        aEntries[44][13] = new CNumber(-93.15, -15.12355);
        aEntries[255][255] = new CNumber(40.1);
        aEntries[304][34] = new CNumber(8);
        aEntries[34][304] = new CNumber(8);
        aEntries[304][55] = new CNumber(345.4, -88.3);
        aEntries[55][304] = new CNumber(345.4, 88.3);
        aEntries[400][400] = new CNumber(0);
        aEntries[402][0] = new CNumber(9.3, 15);
        aEntries[0][402] = new CNumber(9.3, -15);

        buildFromDense(false);

        assertTrue(A.isHermitian());
    }
}
