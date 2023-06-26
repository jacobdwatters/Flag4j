package com.flag4j.linalg;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.io.PrintOptions;
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
        // TODO: Delete
        PrintOptions.setPrecision(50);

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
                {0.002536286923471442, -0.9776911995501422, 0.027427520796710778},
                {0.03778044483704439, -0.15727996676951206, -1.0230636416849341},
                {-0.9992828454629633, -0.00842785658870904, -0.038609924699795885}
        };
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getColSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{
                {new CNumber(0.9768558885215766, -0.20700564109569586),
                        new CNumber(0, 0.981354052294482)},
                {new CNumber(),
                        new CNumber(-0.01201658023217733 ,1.0013816860147775)}};
        expB = new CMatrix(expBEntries);

        assertEquals(expB, SubSpace.getColSpace(B));
    }


    @Test
    void rowSpaceTestCase() {
        // -------------------- Sub-case 1 --------------------
        expAEntries = new double[][]{
                {0.999488309423674, -0.016633076463508676, -0.02332710248931722},
                {-0.01639076956075033, -0.9873715837113857, 0.08329758444462076},
                {-0.02746433647386676, -0.01841362461143939, -0.9049333202771841}};
        expA = new Matrix(expAEntries);

        assertEquals(expA, SubSpace.getRowSpace(A));

        // -------------------- Sub-case 2 --------------------
        expBEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(1)}};
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
