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
                {-0.0025362869234714426, 0.9674946208341382, 5.831001640296924E-9},
                {-0.037780444837044395, 0.25260644036275337, 5.914301663729737E-8},
                {0.9992828454629633, 0.012006037825457215, 2.08250058582033E-10}};
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
                {-0.9994883094236743, 0.017904410553719628, 0.01713906194257628},
                {0.016390769560750335, 0.9397111048050247, 0.24945037570556278},
                {0.027464336473866764, 0.09571327516163725, 0.4619986954434839}};
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
                {0.020218901153584195},
                {-0.23334286412847316},
                {0.8812728774837011},
                {0.4104880256573392}};
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
