package com.flag4j.linalg;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


class SubSpaceTests {

    static double[][] aEntries, expAEntries;
    static CNumber[][] bEntries, expBEntries;

    static Matrix A, expA;
    static CMatrix B, expB;


    @BeforeAll
    static void setup() {
        aEntries = new double[][]{
                {2.4, 5.6, -0.35, 1.5},
                {34.6, 0, 2.4, 2},
                {-912.5, 15, 25.2, -0.3}};

        bEntries = new CNumber[][]{
                {new CNumber(44.5, -9.43), new CNumber(0, 2.45)},
                {new CNumber(), new CNumber(-0.03, 2.5)}};

        A = new Matrix(aEntries);
        B = new CMatrix(bEntries);
    }


    @Test
    void colSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {-0.002536286923471442, -0.9776911995501422, -0.027427520796710778},
                {-0.03778044483704439, -0.15727996676951206, 1.0230636416849341},
                {0.9992828454629633, -0.00842785658870904, 0.038609924699795885}
        };
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getColSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{
                {new CNumber(-0.9782717373949313, 0.20730567378953263),
                        new CNumber(0, -0.0029606467149292277)},
                {new CNumber(-0.0029034857946125137, 5.789639117442682E-4),
                        new CNumber(-0.011999083504397996, 0.9999236253664997)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getColSpace(B));
    }


    @Test
    void rowSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {-0.999488309423674, -0.016633076463508676, 0.02332710248931722},
                {0.01639076956075033, -0.9873715837113857, -0.08329758444462076},
                {0.02746433647386676, -0.01841362461143939, 0.9049333202771841}};
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getRowSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{
                {new CNumber(-0.9985439511540497), new CNumber(0.011182989469365768, -0.05277232570379392)},
                {new CNumber(0.011182989469365768, 0.05277232570379392), new CNumber(0.9985439511540495)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getRowSpace(B));
    }


    @Test
    void nullSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {-0.014223444808413515},
                {-0.13375429255978216},
                {-0.4242668199409785},
                {0.8954915679875406}};
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getNullSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{{new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getNullSpace(B));
    }


    @Test
    void leftNullSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {0.0},
                {0.0},
                {0.0},
                {0.0}};
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getLeftNullSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{{new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getLeftNullSpace(B));
    }


    @Test
    void hasEqualSpan() {
        // -------------------- Sub-case 1 --------------------
        assertTrue(
                SubSpace.hasEqualSpan(A.getSlice(0, A.numRows, 0, A.numRows),
                        SubSpace.getColSpace(A))
        );

        // -------------------- Sub-case 2 --------------------
        assertTrue(
                SubSpace.hasEqualSpan(B.getSlice(0, B.numRows, 0, B.numRows),
                        SubSpace.getColSpace(B))
        );
    }
}
