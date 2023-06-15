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
                {0.0025362869234714426, -0.9674946208341343, -0.2528788759577717},
                {0.03778044483704439, -0.25260644036276714, 0.9668312284339803},
                {-0.9992828454629634, -0.012006037825457441, 0.035911694741080004}};
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getColSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{
                {new CNumber(-0.9782717373949313, 0.20730567378953263),
                        new CNumber(0.0, -0.0029606467149292277)},
                {new CNumber(-0.0029034857946125137, 5.789639117442682E-4),
                        new CNumber(-0.011999083504397996, 0.9999236253664997)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getColSpace(B));
    }


    @Test
    void rowSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {0.999488309423674, -0.017904410553720104, 0.020218901153583838},
                {-0.01639076956075033, -0.9397111048050201, -0.23334286412849414},
                {-0.02746433647386676, -0.09571327516165756, 0.8812728774837002}};
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getRowSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{
                {new CNumber(-0.9985439511540497, 0.0), new CNumber(0.011182989469365768, -0.05277232570379392)},
                {new CNumber(0.011182989469365768, 0.05277232570379392), new CNumber(0.9985439511540495, 0.0)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getRowSpace(B));
    }


    @Test
    void nullSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {0.017139061942576198},
                {0.2494503757055624},
                {0.46199869544348104},
                {-0.8509042061387424}};
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
